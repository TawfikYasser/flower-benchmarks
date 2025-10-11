# task.py  (replace existing file with this)
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import time
import os
import json
import subprocess
from pathlib import Path

# keep your existing small CNN for non-detection tasks
class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -------------------
# Existing classification transforms/data loaders (unchanged)
# -------------------
fds = None  # Cache FederatedDataset
pytorch_transforms = Compose([ToTensor(), Normalize((0.5, ), (0.5, ))])

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch

def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10-like data (existing behavior)."""
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label", alpha=0.7)
        fds = FederatedDataset(
            dataset="zalando-datasets/fashion_mnist",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader

def train(net, trainloader, epochs, lr, partition_id, device):
    """Train the existing small CNN."""
    round_log = {}
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    start = time.perf_counter()
    round_log["round_start_time"] = start
    for _ in range(epochs):
        round_log["client_id"] = partition_id
        round_log["epoch"] = _ + 1
        round_log["lr"] = lr
        start = time.perf_counter()
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (len(trainloader) * epochs)
    end = time.perf_counter()
    round_log["round_end_time"] = end
    round_log["round_duration"] = end - start
    round_log["round_loss"] = avg_trainloss
    return avg_trainloss, round_log

def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

# -------------------
# YOLOv5 detection helpers (NEW)
# -------------------
from flower_benchmarks.plugins.yolov5.model import save_state_dict_as_yolo_checkpoint, load_yolo_checkpoint_as_state_dict, YoloSizeToPretrained
from flower_benchmarks.plugins.yolov5.dataset import partition_coco128_dir, write_client_dataset_yolo_layout, write_data_yaml

def prepare_client_yolo_dataset(global_coco_root: str, tmp_client_root_base: str, client_id: int,
                                num_clients: int, alpha: float = 0.7, seed: int = 42):
    """
    Partition the COCO128 dataset (if not already done) and prepare the client-specific
    YOLO folder (images + labels) and data yaml. Returns path to data yaml.
    """
    os.makedirs(tmp_client_root_base, exist_ok=True)
    partitions = partition_coco128_dir(global_coco_root, num_clients, alpha=alpha, seed=seed)
    client_partition = partitions[client_id]
    client_dataset_root = write_client_dataset_yolo_layout(global_coco_root, tmp_client_root_base, client_id, client_partition)
    # create data yaml
    data_yaml = os.path.join(tmp_client_root_base, f"client_{client_id}", "coco128_client.yaml")
    write_data_yaml(client_dataset_root, data_yaml)
    return data_yaml, client_dataset_root

def yolo_train_from_state_and_return_state_dict(received_state_dict: dict,
                                                model_size: str,
                                                client_dataset_yaml: str,
                                                epochs: int,
                                                img: int = 640,
                                                batch: int = 16,
                                                run_dir: str = "runs/train",
                                                client_tag: str = "client",
                                                round_idx: int = 0,
                                                run_id: str = "1"):
    """
    1. Save the received_state_dict into a YOLO checkpoint file (received_weights.pt).
    2. Run YOLOv5 train.py as a subprocess using that checkpoint as --weights
    3. After finishing, locate the best/last weights and load them, returning a torch state_dict suitable for ArrayRecord.
    Note: runs are saved under run_dir/<client_tag>_round_<round_idx>
    """
    workdir = Path("./")
    tmp_weights = f"received_weights_{client_tag}_r{round_idx}.pt"
    # place weights in run_dir/<name>/weights
    name = f"{client_tag}_r{round_idx}"
    # modify tmp_weights to be in run_dir to avoid clutter
    full_path = os.path.join(run_dir, f"{client_tag}_r{round_idx}", "weights", tmp_weights)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    tmp_weights = full_path
    # save received state dict as YOLO checkpoint
    save_state_dict_as_yolo_checkpoint(received_state_dict, tmp_weights)
    # # prepare command to call YOLOv5 train.py
    # # We assume yolov5 is installed in the environment (pip install git+...).
    # name = f"{client_tag}_r{round_idx}"
    # cmd = [
    #     "python", "-m", "yolov5.train",  # use module style to be robust
    #     "--img", str(img),
    #     "--batch", str(batch),
    #     "--epochs", str(epochs),
    #     "--data", client_dataset_yaml,
    #     "--weights", tmp_weights,
    #     "--project", run_dir,
    #     "--name", name,
    #     "--exist-ok"
    # ]
    # os.environ["PYTHONPATH"] = os.getcwd() + ":" + os.environ.get("PYTHONPATH", "")
    # # If model_size is given as 'n','s',... and weights are standard names, you can pass --cfg or --weights
    # # but we're feeding a starting weights file. To ensure correct arch, user should ensure checkpoint matches size.
    # try:
    #     subprocess.run(cmd, check=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error during YOLOv5 training subprocess: {e}")
    #     print(f"Error output: {e.stderr}")
    #     print(f"Standard output: {e.stdout}")
    #     raise 


    # I want to pass the cfg to the yolo.train based on the specified model n, s, etc.

    cfg = "yolov5s.yaml" if model_size == "s" else "yolov5n.yaml" if model_size == "n" else \
          "yolov5m.yaml" if model_size == "m" else "yolov5l.yaml" if model_size == "l" else \
          "yolov5x.yaml" if model_size == "x" else None
    if cfg is None:
        cfg = "yolov5n.yaml"  # default to small if unknown

    # Ensure project root / yolov5 folder is available for imports at runtime
    os.environ["PYTHONPATH"] = os.getcwd() + ":" + os.environ.get("PYTHONPATH", "")
    import sys, importlib

    # Try in-process import & run
    try:
        # ensure local yolov5 directory is preferred on sys.path (useful if you work with cloned repo)
        yolov5_local = os.path.join(os.getcwd(), "yolov5")
        if os.path.isdir(yolov5_local) and yolov5_local not in sys.path:
            sys.path.insert(0, yolov5_local)


        # Add safe globals for YOLOv5 model loading
        from torch.serialization import add_safe_globals
        from yolov5.models.yolo import DetectionModel
        add_safe_globals([DetectionModel])

        import torch

        orig_load = torch.load

        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return orig_load(*args, **kwargs)

        torch.load = patched_load

        ytrain = importlib.import_module("yolov5.train")
        # call yolov5.train.run(...) directly (no CLI brittleness)
        try:
            ytrain.run(
                data=client_dataset_yaml,
                imgsz=img,
                batch_size=batch,
                epochs=epochs,
                cfg=cfg,
                weights=tmp_weights,
                project=str(run_dir),
                name=name,
                exist_ok=True,
                disable_wandb=True,  # disable wandb logging
            )
            print(f"[yolo_train] In-process yolov5.train.run completed successfully.")
        except Exception as e:
            # In-process training error: show a helpful message and raise
            print(f"[yolo_train] In-process yolov5.train.run raised an exception: {e}")
            raise

    except Exception as import_exc:
        # If we can't import or calling in-process failed, fallback to subprocess (robust)
        print(f"[yolo_train] Could not run in-process (import/call error): {import_exc}")
        print("[yolo_train] Falling back to subprocess call of `python -m yolov5.train`")

        cmd = [
            "python", "-m", "yolov5.train",
            "--img", str(img),
            "--batch-size", str(batch),   # use correct flag name
            "--epochs", str(epochs),
            "--data", client_dataset_yaml,
            "--weights", tmp_weights,
            "--project", run_dir,
            "--name", name,
            "--exist-ok",
            "--cfg", cfg
        ]
        # keep PYTHONPATH for subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")

        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            print("YOLOv5 subprocess training failed.")
            print("=== STDOUT ===")
            print(proc.stdout)
            print("=== STDERR ===")
            print(proc.stderr)
            # Raise a CalledProcessError so upstream code sees a failure (like before)
            raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout, stderr=proc.stderr)
        else:
            print(f"[yolo_train] YOLOv5 subprocess training completed successfully.")

    # try to find best.pt under runs/train/<name>/weights/best.pt or last.pt
    out_dir = Path(run_dir) / name / "weights"
    candidate_best = out_dir / "best.pt"
    candidate_last = out_dir / "last.pt"
    if candidate_best.exists():
        result_ckpt = str(candidate_best)
    elif candidate_last.exists():
        result_ckpt = str(candidate_last)
    else:
        # fallback: pick any .pt in weights subdir
        pt_files = list(out_dir.glob("*.pt"))
        if pt_files:
            result_ckpt = str(pt_files[-1])
        else:
            raise FileNotFoundError(f"No trained weights found in {out_dir}")

    final_state = load_yolo_checkpoint_as_state_dict(result_ckpt)
    return final_state

def yolo_evaluate_weights_and_parse_map(weights_pt: str, data_yaml: str, img: int = 640):
    """
    Evaluate YOLOv5 weights using in-process `yolov5.val.run` when possible.
    Falls back to subprocess if imports fail.
    Returns a dict containing parsed mAP metrics.
    """
    import sys, importlib, os, subprocess

    metrics = {}

    # Ensure PYTHONPATH includes current working directory and yolov5
    os.environ["PYTHONPATH"] = os.getcwd() + ":" + os.environ.get("PYTHONPATH", "")
    yolov5_local = os.path.join(os.getcwd(), "yolov5")

    # Add yolov5 to sys.path if needed
    if os.path.isdir(yolov5_local) and yolov5_local not in sys.path:
        sys.path.insert(0, yolov5_local)

    try:
        # Ensure safe torch deserialization
        from torch.serialization import add_safe_globals
        from yolov5.models.yolo import DetectionModel
        add_safe_globals([DetectionModel])

        import torch
        orig_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return orig_load(*args, **kwargs)
        torch.load = patched_load

        # Import YOLOv5 val module
        yval = importlib.import_module("yolov5.val")

        # Run in-process validation
        print("[yolo_eval] Running yolov5.val.run in-process...")
        results = yval.run(
            weights=weights_pt,
            data=data_yaml,
            imgsz=img,
            task='val',
            verbose=True
        )

        # Try to extract metrics from YOLOv5 results dict (YOLOv5 >= 7.x style)
        if isinstance(results, dict):
            if "metrics" in results:
                metrics = results["metrics"]
            elif "mAP@0.5" in results:
                metrics = {"mAP@0.5": results["mAP@0.5"]}
            else:
                # fallback if structure differs
                for k, v in results.items():
                    if isinstance(v, (float, int)) and "map" in k.lower():
                        metrics[k] = v

        print(f"[yolo_eval] In-process evaluation completed successfully. Metrics: {metrics}")

    except Exception as e:
        print(f"[yolo_eval] In-process validation failed.")
        print("[yolo_eval] Falling back to subprocess mode...")

        # Fallback subprocess call
        cmd = [
            "python", "-m", "yolov5.val",
            "--weights", weights_pt,
            "--data", data_yaml,
            "--img", str(img)
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

        stdout = proc.stdout
        if proc.returncode != 0:
            print("YOLOv5 subprocess evaluation failed.")
            print("=== STDOUT ===")
            print(stdout)
            print("=== STDERR ===")
            print(proc.stderr)
            raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout, stderr=proc.stderr)

        # Parse mAP lines from stdout
        for line in stdout.splitlines():
            if "mAP@0.5" in line:
                try:
                    toks = [t for t in line.replace(",", " ").split() if any(ch.isdigit() for ch in t)]
                    for t in toks:
                        try:
                            val = float(t)
                            metrics["mAP@0.5"] = val
                            break
                        except:
                            pass
                except Exception:
                    pass

        print(f"[yolo_eval] Subprocess evaluation completed. Metrics: {metrics}")

    return metrics
