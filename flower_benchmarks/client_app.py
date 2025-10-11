"""flower-benchmarks: A Flower / PyTorch app."""

import os
import torch
import numpy as np
import sys
import pickle
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flower_benchmarks.task import Net, load_data, test as test_fn, train as train_fn
from flower_benchmarks.plugins.yolov5.model import load_yolo_checkpoint_as_state_dict
from flower_benchmarks.task import yolo_train_from_state_and_return_state_dict, prepare_client_yolo_dataset, yolo_evaluate_weights_and_parse_map
from pathlib import Path

# Flower ClientApp
app = ClientApp()

def calculate_message_size(msg: Message) -> dict:
    """Calculate size of different components in a message."""
    sizes = {}
    
    # Calculate arrays size (model parameters)
    if "arrays" in msg.content:
        arrays_size = 0
        state_dict = msg.content["arrays"].to_torch_state_dict()
        for param in state_dict.values():
            arrays_size += param.nelement() * param.element_size()
        sizes["arrays_bytes"] = arrays_size
    
    # Calculate metrics size
    if "metrics" in msg.content:
        metrics_size = sys.getsizeof(pickle.dumps(dict(msg.content["metrics"])))
        sizes["metrics_bytes"] = metrics_size
    
    # Calculate config size
    if "config" in msg.content:
        config_size = sys.getsizeof(pickle.dumps(dict(msg.content["config"])))
        sizes["config_bytes"] = config_size
    
    sizes["total_bytes"] = sum(sizes.values())
    return sizes["total_bytes"] # Return total size for simplicity

def extract_yolov5_weights_as_arrays(model):
    """
    Extract YOLOv5 model weights and convert to a proper format for ArrayRecord.
    
    Args:
        model: Trained YOLOv5 model object
        
    Returns:
        Dictionary of NumPy arrays or PyTorch state_dict
    """
    # Access the underlying PyTorch model
    if hasattr(model, 'model'):
        pytorch_model = model.model
    else:
        pytorch_model = model
    
    # Get the state_dict (PyTorch format)
    state_dict = pytorch_model.state_dict()
    
    # Ensure all values are properly formatted as NumPy arrays
    weights_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # Convert tensor to NumPy
            weights_dict[key] = value.cpu().detach().numpy()
        elif isinstance(value, np.ndarray):
            weights_dict[key] = value
        else:
            # Handle any other types by converting to numpy
            weights_dict[key] = np.array(value)
    
    return weights_dict

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data. Supports two modes:
       - classification (default): uses Net() and the original train()
       - detection: uses YOLO plugin when context.run_config['task']=='detection'
    """
    # Calculate size of received message from server
    received_sizes = calculate_message_size(msg)

    task_type = context.run_config.get("task", "classification")

    if task_type == "detection":
        # YOLO detection flow
        # Save incoming arrays to checkpoint, call YOLO train via helper, then load new weights and return them.
        received_state = msg.content["arrays"].to_torch_state_dict()
        # prepare client-specific dataset
        # global path to coco128
        coco_root = context.run_config.get("coco128_root")  # Relative to flower_benchmarks directory
        tmp_clients_base = context.run_config.get("yolo_tmp_dir")
        num_clients = context.node_config.get("num-partitions", 1)
        client_id = context.node_config.get("partition-id", 0)
        # prepare dataset for this client
        data_yaml, client_dataset_root = prepare_client_yolo_dataset(coco_root, tmp_clients_base, client_id, num_clients,
                                                                     alpha=context.run_config.get("dirichlet_alpha", 0.7),
                                                                     seed=context.run_config.get("dirichlet_seed", 42))
        # model size (n/s/m/l/x)
        model_size = context.run_config.get("yolo_size", "n")
        # training args
        epochs = context.run_config.get("local-epochs", 1)
        img = context.run_config.get("img_size", 640)
        batch = context.run_config.get("batch_size", 16)
        run_id = str(context.run_config.get("run_id", "1"))
        # tag/round info
        num_rounds = msg.content["config"].get("num_rounds", 0)
        server_round = msg.content["config"].get("server-round", 0)
        # train using YOLO CLI called by helper
        new_state = yolo_train_from_state_and_return_state_dict(received_state,
                                                                model_size=model_size,
                                                                client_dataset_yaml=data_yaml,
                                                                epochs=epochs,
                                                                img=img,
                                                                batch=batch,
                                                                run_dir=context.run_config.get("yolo_runs_dir"),
                                                                client_tag=f"client{client_id}",
                                                                round_idx=server_round,
                                                                run_id=run_id)

        # Construct reply Message
        # Extract YOLOv5 weights as a proper dictionary of NumPy arrays
        final_weights = list(extract_yolov5_weights_as_arrays(new_state).values())
        model_record = ArrayRecord(final_weights, keep_input=True)

        round_log = {"client_id": client_id, "num_rounds": num_rounds, "server_round_number": server_round,
                     "data_received_from_server": received_sizes}
        metrics = {"num-examples": len(list(Path(client_dataset_root).glob('images/train2017/*.jpg'))),
                   **round_log}
        metric_record = MetricRecord(metrics)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        reply_msg = Message(content=content, reply_to=msg)
        sent_sizes = calculate_message_size(reply_msg)
        metrics["data_sent_to_server"] = sent_sizes
        return reply_msg
    

    else:
        # existing classification flow (unchanged)
        model = Net()
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        device = torch.device("cpu")
        model.to(device)
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        trainloader, _ = load_data(partition_id, num_partitions)
        train_loss, round_log = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            partition_id,
            device,
        )
        round_log["num_rounds"] = msg.content["config"]["num_rounds"]
        round_log["server_round_number"] = msg.content["config"]["server-round"]
        round_log["data_received_from_server"] = received_sizes
        model_record = ArrayRecord(model.state_dict())
        metrics = {
            "train_loss": train_loss,
            "num-examples": len(trainloader.dataset),
            **round_log
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        reply_msg = Message(content=content, reply_to=msg)
        sent_sizes = calculate_message_size(reply_msg)
        metrics["data_sent_to_server"] = sent_sizes
        return reply_msg


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    task_type = context.run_config.get("task", "classification")

    if task_type == "detection":
        # Consider that client val images already prepared in train()

        # Load the data
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        tmp_clients_base = context.run_config.get("yolo_tmp_dir")
        client_id = context.node_config.get("partition-id", 0)
        client_dataset_root = os.path.join(tmp_clients_base, f"client_{client_id}")
        val_yaml = os.path.join(client_dataset_root, "coco128_client.yaml")

        # Load the model and initialize it with the received weights
        trained_model = msg.content["arrays"].to_torch_state_dict()

        # Evaluate the returned weights
        # save temporary ckpt to disk to run val
        tmp_out_ckpt = os.path.join(context.run_config.get("yolo_runs_dir", "runs/train"),
                                    f"client{client_id}_r{context.run_config['server-round']}",
                                    "weights", f"client{client_id}_r{context.run_config['server-round']}_for_val.pt")
        torch.save({"model": trained_model}, tmp_out_ckpt)
        load_yolo_checkpoint_as_state_dict(tmp_out_ckpt)  # verify load works

        # Call the evaluation function
        val_metrics = yolo_evaluate_weights_and_parse_map(tmp_out_ckpt, val_yaml, img=context.run_config.get("img_size", 640))

        # Construct and return reply Message
        metrics = {
            # "eval_loss": eval_loss,
            # "eval_acc": eval_acc,
            **val_metrics,
            "num-examples": len(list(Path(client_dataset_root).glob('images/val2017/*.jpg'))),
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict({"metrics": metric_record})
        return Message(content=content, reply_to=msg)

    else:

        # Load the model and initialize it with the received weights
        model = Net()
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load the data
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        _, valloader = load_data(partition_id, num_partitions)

        # Call the evaluation function
        eval_loss, eval_acc = test_fn(
            model,
            valloader,
            device,
        )

        # Construct and return reply Message
        metrics = {
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "num-examples": len(valloader.dataset),
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict({"metrics": metric_record})
        return Message(content=content, reply_to=msg)
