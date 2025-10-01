"""flower-benchmarks: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict, ConfigRecord
from flwr.clientapp import ClientApp

from flower_benchmarks.task import Net, load_data
from flower_benchmarks.task import test as test_fn
from flower_benchmarks.task import train as train_fn
import sys
import pickle

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


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

     # Calculate size of received message from server
    received_sizes = calculate_message_size(msg)

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions)

    # Call the training function
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
    # Add data transmission info
    round_log["data_received_from_server"] = received_sizes

    # Construct reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        **round_log
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    reply_msg = Message(content=content, reply_to=msg)
    
    # Calculate size of message being sent back to server
    sent_sizes = calculate_message_size(reply_msg)
    metrics["data_sent_to_server"] = sent_sizes
    return reply_msg


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

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
