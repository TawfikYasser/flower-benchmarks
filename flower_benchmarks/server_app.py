"""flower-benchmarks: A Flower / PyTorch app."""

import torch
from typing import List, Tuple, Dict, Optional
from flwr.app import ArrayRecord, Context, MetricRecord, RecordDict, ConfigRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from flower_benchmarks.task import Net
import logging
import os
import json
import shutil

# Global variable to store round logs from all rounds
ALL_ROUND_LOGS = []

# Create ServerApp
app = ServerApp()


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def custom_train_metrics_aggregation(record_dicts: List[RecordDict], weighted_by_key: str) -> MetricRecord:
    """Collect per-client training metrics for this round."""
    global ALL_ROUND_LOGS

    if not record_dicts:
        return MetricRecord({})

    current_round_logs = []
    total_data_server_to_clients = 0.0
    total_data_clients_to_server = 0.0

    for record_dict in record_dicts:
        if "metrics" in record_dict:
            round_log_data = {}
            metrics = record_dict["metrics"]
            # metrics may be a mapping-like object
            for key, value in metrics.items():
                # skip raw train_loss recording here (we can add if needed)
                if key == "train_loss":
                    continue
                round_log_data[key] = value

                # Aggregate transmission data (if present and numeric)
                if key == "data_received_from_server":
                    total_data_server_to_clients += _safe_float(value)
                elif key == "data_sent_to_server":
                    total_data_clients_to_server += _safe_float(value)

            # Remove large or detailed transmission fields if present (safe pop)
            round_log_data.pop("data_received_from_server", None)
            round_log_data.pop("data_sent_to_server", None)

            current_round_logs.append(round_log_data)

    # Calculate total round trip data (in bytes). Convert to MB for easier human reading.
    total_round_data = total_data_server_to_clients + total_data_clients_to_server
    total_round_data_mb = round(total_round_data / (1024 ** 2), 4)

    # Append summary for this round
    ALL_ROUND_LOGS.append(
        {
            "clients_logs": current_round_logs,
            "total_amount_data_round_mb": total_round_data_mb,
        }
    )

    # We intentionally return an empty MetricRecord here because per-client train metrics
    # are stored in ALL_ROUND_LOGS for later analysis. If you want to return aggregated
    # train metrics to the strategy, add keys here.
    return MetricRecord({})


def custom_eval_metrics_aggregation(record_dicts: List[RecordDict], weighted_by_key: str) -> MetricRecord:
    """
    Aggregate client evaluation accuracies into a global round accuracy and append to logs.
    Supports detection metrics: clients may send 'eval_acc' OR 'mAP@0.5' OR 'mAP'.
    Uses 'num-examples' as weights. Returns a MetricRecord containing aggregated metrics.
    """
    global ALL_ROUND_LOGS

    total_weighted_score = 0.0
    total_examples = 0.0

    # Keep track of raw metrics for debug if needed
    per_client_debug = []

    for record_dict in record_dicts:
        if "metrics" not in record_dict:
            continue
        metrics = record_dict["metrics"]

        # Determine candidate accuracy/mAP value from common keys
        # Priority: eval_acc -> mAP@0.5 -> mAP
        acc_value = None
        if "eval_acc" in metrics:
            acc_value = _safe_float(metrics["eval_acc"])
        elif "mAP@0.5" in metrics:
            acc_value = _safe_float(metrics["mAP@0.5"])
        elif "mAP" in metrics:
            acc_value = _safe_float(metrics["mAP"])
        # If still None, skip this client's contribution
        if acc_value is None:
            continue

        # Determine number of examples (weight)
        n_examples = 0.0
        if "num-examples" in metrics:
            n_examples = _safe_float(metrics["num-examples"])
        else:
            # Fallback: some clients may send "num_examples" or "num_examples_train" etc.
            n_examples = _safe_float(metrics.get("num_examples", 0.0))

        # Accumulate
        total_weighted_score += acc_value * n_examples
        total_examples += n_examples

        per_client_debug.append({"acc": acc_value, "n": n_examples})

    # Compute aggregated accuracy/mAP (handle zero examples)
    aggregated_score = total_weighted_score / total_examples if total_examples > 0 else 0.0

    # If we store logs per round, append aggregated score (as percentage for easier reading)
    if ALL_ROUND_LOGS:
        # Store percentage rounded to 2 decimals
        try:
            ALL_ROUND_LOGS[-1]["round_acc"] = round(aggregated_score * 100, 2)
            ALL_ROUND_LOGS[-1]["per_client_eval_debug"] = per_client_debug
        except Exception:
            # Silently ignore logging errors to avoid breaking aggregation
            pass

    # Return numeric aggregated metric (not percent) so downstream code can interpret correctly
    return MetricRecord({"round_acc": aggregated_score})


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config (use .get to provide defaults if absent)
    fraction_train: float = context.run_config.get("fraction-train", 1.0)
    fraction_evaluate: float = context.run_config.get("fraction_evaluate", 1.0)
    num_rounds: int = int(context.run_config.get("num-server-rounds", 5))
    lr: float = float(context.run_config.get("lr", 0.01))
    task_type: str = context.run_config.get("task", "classification")

    # Choose an experiment name depending on task
    if task_type == "detection":
        experiment_name: str = context.run_config.get("experiment_name",
                                                      f"EXP_YOLOv5_{context.run_config.get('yolo_size', 'n')}_detection")
    else:
        experiment_name: str = context.run_config.get("experiment_name", "EXP_CNN_fashion_mnist_dataset")

    # Get run id interactively (keeps previous behavior)
    run_id = str(context.run_config.get("run_id", "1"))

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy with our custom metric aggregation hooks
    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        train_metrics_aggr_fn=custom_train_metrics_aggregation,
        evaluate_metrics_aggr_fn=custom_eval_metrics_aggregation,
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr, "num_rounds": num_rounds}),
        num_rounds=num_rounds,
    )

    # Save final model to disk
    state_dict = result.arrays.to_torch_state_dict()

    if task_type == "detection":
        # Save in YOLO-friendly checkpoint format (wrap in 'model' key)
        out_path = f"{experiment_name}_{run_id}_final_model.pt"
        torch.save({"model": state_dict}, out_path)
    else:
        out_path = f"{experiment_name}_{run_id}_final_model.pt"
        torch.save(state_dict, out_path)

    # Save round logs to file (JSON)
    logs_path = f"{experiment_name}_{run_id}_logs.json"
    try:
        with open(logs_path, "w") as f:
            json.dump(ALL_ROUND_LOGS, f, indent=2)
    except Exception as exc:
        pass

    # Copy the analysis notebook to a new file with the run_id in its name (if exists)
    src_notebook = "analysis.ipynb"
    dst_notebook = f"run_{run_id}_benchmarks.ipynb"
    if os.path.exists(src_notebook):
        try:
            shutil.copy(src_notebook, dst_notebook)

            # Replace the placeholder log filename inside the notebook if present
            try:
                with open(dst_notebook, "r", encoding="utf-8") as f:
                    notebook_content = f.read()
                # Replace a common default name if present. Use a safe replace.
                updated_content = notebook_content.replace("EXP_CNN_fashion_mnist_dataset_1_logs.json", logs_path)
                with open(dst_notebook, "w", encoding="utf-8") as f:
                    f.write(updated_content)
            except Exception as exc:
                pass
        except Exception as exc:
            pass
    else:
        pass

    print(f"Run completed. Final model saved to {out_path}, logs saved to {logs_path}.")
