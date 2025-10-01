"""flower-benchmarks: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, MetricRecord, RecordDict, ConfigRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from flower_benchmarks.task import Net

# Global variable to store round logs from all rounds
ALL_ROUND_LOGS = []

# Create ServerApp
app = ServerApp()

def custom_train_metrics_aggregation(record_dicts: list[RecordDict], weighted_by_key: str) -> MetricRecord:
    """Collect per-client training metrics for this round."""
    global ALL_ROUND_LOGS
    
    if not record_dicts:
        return MetricRecord({})
    
    current_round_logs = []
    total_data_server_to_clients = 0
    total_data_clients_to_server = 0
    
    for record_dict in record_dicts:
        if "metrics" in record_dict:            
            round_log_data = {}
            for key, value in record_dict["metrics"].items():
                if key != "train_loss":
                    round_log_data[key] = value
                    
                    # Aggregate transmission data
                    if key == "data_received_from_server":
                        total_data_server_to_clients += value
                    elif key == "data_sent_to_server":
                        total_data_clients_to_server += value
            del round_log_data["data_received_from_server"]  # Remove detailed data to reduce log size
            current_round_logs.append(round_log_data)
    
    # Calculate total round trip data
    total_round_data = total_data_server_to_clients + total_data_clients_to_server
    
    # Add a new round entry with transmission summary
    ALL_ROUND_LOGS.append({
        "clients_logs": current_round_logs,
        # "server_to_clients_mb": round(total_data_server_to_clients / (1024**2), 4),
        # "clients_to_server_mb": round(total_data_clients_to_server / (1024**2), 4),
        "total_amount_data_round_mb": round(total_round_data / (1024**2), 4)
    })
    
    return MetricRecord({})
def custom_eval_metrics_aggregation(record_dicts: list[RecordDict], weighted_by_key: str) -> MetricRecord:
    """Aggregate client evaluation accuracies into a global round accuracy and append to logs."""
    global ALL_ROUND_LOGS
    
    total_correct = 0.0
    total_examples = 0

    for record_dict in record_dicts:
        if "metrics" in record_dict:
            metrics = record_dict["metrics"]
            if "eval_acc" in metrics and "num-examples" in metrics:
                acc = metrics["eval_acc"]
                n = metrics["num-examples"]
                total_correct += acc * n
                total_examples += n

    round_acc = total_correct / total_examples if total_examples > 0 else 0.0

    # Append round_acc to the last round’s log (same round, same clients)
    if ALL_ROUND_LOGS:
        ALL_ROUND_LOGS[-1]["round_acc"] = round(round_acc*100, 2)

    return MetricRecord({"round_acc": round_acc})

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    experiment_name: str = "EXP1_CNN_fashion_mnist"
    run_id: str = input('Type the Run ID: ')

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=fraction_train,
                      train_metrics_aggr_fn=custom_train_metrics_aggregation,
                      evaluate_metrics_aggr_fn=custom_eval_metrics_aggregation)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr, "num_rounds": num_rounds}),
        num_rounds=num_rounds,
    )
    
    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, f"{experiment_name}_{run_id}_final_model.pt")

    # Save round logs to file
    import json
    with open(f"{experiment_name}_{run_id}_logs.json", "w") as f:
        json.dump(ALL_ROUND_LOGS, f, indent=2)
    print("Logs saved to disk")