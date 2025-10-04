# Federated Learning Benchmarks: Fashion-MNIST CNN with Flower Framework

A production-ready federated learning implementation using Flower framework for image classification on Fashion-MNIST dataset, featuring comprehensive performance tracking, communication overhead analysis, and experiment reproducibility tools.

---

## 🎯 Overview

This project demonstrates **privacy-preserving federated learning** for image classification using a simple CNN architecture trained on the Fashion-MNIST dataset. Built with the Flower framework, it provides detailed benchmarking capabilities to analyze training efficiency, communication costs, and model performance across federated rounds.

### Key Highlights

- **Flower Framework Integration**: Modern federated learning with simulation and production deployment support
- **Privacy-Preserving**: Train models across distributed clients without sharing raw data
- **Comprehensive Benchmarking**: Track communication overhead, training metrics, and model performance
- **Non-IID Data Distribution**: Dirichlet partitioning (α=0.3) simulates realistic heterogeneous data scenarios
- **Production-Ready**: JSON logging, model checkpointing, and experiment tracking built-in

---

## ✨ Features

### Core Functionality
- ✅ **Flower Framework**: Latest Flower 1.22.0+ with simulation support
- ✅ **Simple CNN Architecture**: Lightweight model optimized for Fashion-MNIST
- ✅ **Non-IID Partitioning**: Dirichlet distribution (α=0.3) for realistic federated scenarios
- ✅ **Configurable Clients**: Simulate 1-50+ SuperNodes with flexible configuration
- ✅ **FedAvg Strategy**: Federated averaging with custom metric aggregation

### Training & Optimization
- ✅ **Adam Optimizer**: Efficient training with configurable learning rates
- ✅ **GPU/CPU Support**: Automatic device detection and utilization
- ✅ **Local Epochs**: Configurable local training iterations per round
- ✅ **Model Checkpointing**: Automatic final model saving with experiment tracking
- ✅ **Train-Test Split**: 80/20 split on each client for local validation

### Advanced Benchmarking Features

#### 📊 Communication Analysis
- **Message Size Tracking**: Precise byte-level measurement of:
  - Model parameters (arrays)
  - Metrics payload
  - Configuration data
  - Total round-trip data transfer
- **Per-Round Aggregation**: Total data transmitted in MB per communication round
- **Bandwidth Optimization**: Identify communication bottlenecks

#### 📈 Performance Metrics
- **Training Metrics**: 
  - Loss per client per round
  - Training duration per client
  - Number of local examples
- **Evaluation Metrics**:
  - Round accuracy (weighted average across clients)
  - Per-client evaluation loss and accuracy
- **Timing Analysis**:
  - Round start/end timestamps
  - Round duration tracking
  - Training speed analysis

#### 🔬 Experiment Tracking
- **JSON Logging**: Structured logs with:
  - Per-client training logs
  - Communication overhead summary
  - Round accuracy progression
  - Complete experiment reproducibility
- **Run Identification**: Unique run IDs for experiment management
- **Model Artifacts**: Saved final model with experiment metadata

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- pip or conda package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/flower-benchmarks.git
cd flower-benchmarks
```

### Step 2: Install Dependencies

#### Using pip (recommended):
```bash
pip install -e .
```

This will install all dependencies specified in `pyproject.toml`:
- `flwr[simulation]>=1.22.0` - Flower framework with simulation support
- `flwr-datasets[vision]>=0.5.0` - Federated datasets with vision utilities
- `torch==2.7.1` - PyTorch deep learning framework
- `torchvision==0.22.1` - Vision utilities and transforms

#### Using conda (alternative):
```bash
conda create -n flower-bench python=3.10
conda activate flower-bench
pip install -e .
```

### Step 3: Verify Installation
```bash
flwr --version
```

---

## 🚀 Usage

### Quick Start: Local Simulation

Run a federated learning simulation with default configuration (50 clients, 3 rounds):

```bash
flwr run
```

### Custom Configuration

#### Modify Number of Clients:
Edit `pyproject.toml`:
```toml
[tool.flwr.federations.local-simulation]
options.num-supernodes = 20  # Change to desired number of clients
```

#### Modify Training Parameters:
Edit `pyproject.toml`:
```toml
[tool.flwr.app.config]
num-server-rounds = 5    # Number of federated rounds
local-epochs = 2         # Local training epochs per round
lr = 0.001               # Learning rate
fraction-train = 1.0     # Fraction of clients to sample per round
```

### Run with Custom Settings:
```bash
flwr run --run-config "num-server-rounds=10 local-epochs=3 lr=0.001"
```

### Production Deployment

For real distributed deployment with SuperLink:

1. **Configure Remote Federation**:
Edit `pyproject.toml`:
```toml
[tool.flwr.federations.remote-federation]
address = "your-superlink-address:9092"
insecure = false
root-certificates = "/path/to/ca.crt"
```

2. **Run ServerApp**:
```bash
flwr run --federation remote-federation
```

3. **Connect ClientApps** (on client machines):
```bash
flwr-client --insecure --server your-superlink-address:9092
```

---

## ⚙️ Configuration

### Project Structure
```toml
# pyproject.toml

[tool.flwr.app.config]
num-server-rounds = 3    # Total federated learning rounds
fraction-train = 1       # Fraction of clients per round (1.0 = all clients)
local-epochs = 1         # Local training epochs on each client
lr = 0.01                # Learning rate for Adam optimizer

[tool.flwr.federations.local-simulation]
options.num-supernodes = 50  # Number of simulated clients
```

### Key Configuration Options

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `num-server-rounds` | Total communication rounds | 3 | 1-100+ |
| `local-epochs` | Epochs per client per round | 1 | 1-10 |
| `lr` | Learning rate | 0.01 | 0.0001-0.1 |
| `fraction-train` | Client sampling rate | 1.0 | 0.0-1.0 |
| `num-supernodes` | Number of clients | 50 | 1-1000+ |

---

## 📊 Benchmarking & Logging

### Comprehensive Logging System

The project tracks detailed metrics throughout training:

#### Training Logs (Per Client, Per Round)
```json
{
  "clients_logs": [
    {
      "client_id": 0,
      "epoch": 1,
      "lr": 0.01,
      "round_start_time": 1234567890.123,
      "round_end_time": 1234567895.456,
      "round_duration": 5.333,
      "round_loss": 0.456,
      "num-examples": 1200,
      "num_rounds": 3,
      "server_round_number": 1,
      "data_sent_to_server": 4567890
    }
  ],
  "total_amount_data_round_mb": 218.45,
  "round_acc": 87.34
}
```

#### Key Metrics Tracked

1. **Communication Overhead**
   - `data_received_from_server`: Bytes received from server (model + config)
   - `data_sent_to_server`: Bytes sent to server (updated model + metrics)
   - `total_amount_data_round_mb`: Total round-trip data in MB

2. **Training Performance**
   - `round_loss`: Average training loss for the round
   - `round_duration`: Time spent training (seconds)
   - `num-examples`: Number of training samples on client

3. **Model Performance**
   - `round_acc`: Weighted average accuracy across all clients (%)
   - `eval_loss`: Evaluation loss on local test set
   - `eval_acc`: Evaluation accuracy on local test set

### Output Files

After training completes, two files are generated:

1. **Model Checkpoint**: `{experiment_name}_{run_id}_final_model.pt`
   - Final global model state dictionary
   - Ready for deployment or further analysis

2. **Experiment Logs**: `{experiment_name}_{run_id}_logs.json`
   - Complete training history
   - Per-round client logs
   - Communication overhead statistics
   - Round accuracy progression

### Example Log Analysis

```python
import json
import pandas as pd

# Load logs
with open('EXP1_CNN_fashion_mnist_run001_logs.json', 'r') as f:
    logs = json.load(f)

# Extract round accuracies
round_accs = [round_log['round_acc'] for round_log in logs]
print(f"Accuracy progression: {round_accs}")

# Extract communication overhead
comm_data = [round_log['total_amount_data_round_mb'] for round_log in logs]
print(f"Total data transferred: {sum(comm_data):.2f} MB")

# Analyze per-client training times
client_durations = []
for round_log in logs:
    for client in round_log['clients_logs']:
        client_durations.append(client['round_duration'])
print(f"Average training time per client: {sum(client_durations)/len(client_durations):.2f}s")
```

---

## 📁 Project Structure

```
flower-benchmarks/
│
├── flower_benchmarks/
│   ├── __init__.py
│   ├── client_app.py          # ClientApp with benchmarking
│   ├── server_app.py           # ServerApp with custom aggregation
│   └── task.py                 # Model, data loading, train/test functions
│
├── pyproject.toml              # Project configuration and dependencies
├── README.md                   # This file
│
└── outputs/                    # Generated after training
    ├── EXP1_CNN_fashion_mnist_{run_id}_final_model.pt
    └── EXP1_CNN_fashion_mnist_{run_id}_logs.json
```

---

## 🔬 Technical Details

### Model Architecture

Simple CNN adapted for Fashion-MNIST (28×28 grayscale images, 10 classes):

```
Conv2D(1→6, 5×5) → ReLU → MaxPool(2×2)
Conv2D(6→16, 5×5) → ReLU → MaxPool(2×2)
Flatten
FC(256→120) → ReLU
FC(120→84) → ReLU
FC(84→10) → Softmax
```

**Parameters**: ~44K trainable parameters  
**Input**: 28×28×1 grayscale images  
**Output**: 10 class probabilities

### Data Partitioning

- **Dataset**: Fashion-MNIST (70,000 images)
- **Partitioner**: Dirichlet with α=0.3 (high heterogeneity)
- **Per-Client Split**: 80% train, 20% test
- **Non-IID Distribution**: Simulates realistic federated scenarios

### Communication Protocol

1. **Server → Client**: 
   - Model parameters (state_dict)
   - Configuration (learning rate, round number)
   
2. **Client → Server**:
   - Updated model parameters
   - Training metrics (loss, duration, examples)
   - Communication statistics

3. **Message Size Calculation**:
   - Arrays: `sum(param.nelement() * param.element_size())`
   - Metrics: `sys.getsizeof(pickle.dumps(metrics_dict))`
   - Config: `sys.getsizeof(pickle.dumps(config_dict))`

---

## 🎓 Use Cases

### Research Applications
- Federated learning algorithm benchmarking
- Communication efficiency studies
- Non-IID data distribution analysis
- Privacy-preserving ML research
- Convergence analysis under data heterogeneity

### Educational Applications
- Teaching federated learning concepts
- Demonstrating privacy-preserving ML
- Hands-on FL experimentation
- Algorithm comparison studies

### Industrial Prototyping
- Cross-silo federated learning POCs
- Edge device ML scenarios
- Privacy-compliant model training
- Distributed learning system design

---

## 📈 Performance Analysis Tools

### Built-in Metrics

The benchmarking system provides ready-to-analyze metrics:

```python
# Example: Plot accuracy over rounds
import matplotlib.pyplot as plt
import json

with open('logs.json', 'r') as f:
    logs = json.load(f)

rounds = range(1, len(logs) + 1)
accs = [r['round_acc'] for r in logs]

plt.plot(rounds, accs, marker='o')
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.title('Federated Learning Convergence')
plt.grid(True)
plt.show()
```

### Communication Overhead Analysis

```python
# Total data transmitted per round
comm_overhead = [r['total_amount_data_round_mb'] for r in logs]
print(f"Total communication: {sum(comm_overhead):.2f} MB")
print(f"Average per round: {sum(comm_overhead)/len(comm_overhead):.2f} MB")
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: Import errors for `flwr` or `flwr_datasets`
- **Solution**: Run `pip install -e .` from project root

**Issue**: CUDA out of memory
- **Solution**: Code automatically falls back to CPU; reduce batch size in `task.py` if needed

**Issue**: Dataset download fails
- **Solution**: `flwr-datasets` downloads automatically; check internet connection

**Issue**: Logs not saving
- **Solution**: Ensure write permissions in project directory

### Debug Mode

Enable verbose Flower logging:
```bash
export FLWR_TELEMETRY_ENABLED=0  # Disable telemetry
flwr run --verbose
```

---

## 🚀 Advanced Usage

### Experiment Tracking

Integrate with experiment tracking tools:

```python
# In server_app.py, add WandB integration
import wandb

wandb.init(project="flower-benchmarks", name=f"run_{run_id}")

# Log metrics after each round
for round_log in ALL_ROUND_LOGS:
    wandb.log({
        "round_acc": round_log['round_acc'],
        "comm_overhead_mb": round_log['total_amount_data_round_mb']
    })
```

### Custom Strategies

Implement custom aggregation strategies:

```python
from flwr.serverapp.strategy import Strategy

class CustomStrategy(Strategy):
    def aggregate_fit(self, server_round, results, failures):
        # Your custom aggregation logic
        pass
```

### Hyperparameter Optimization

Use Flower with Optuna for HPO:

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    # Run Flower experiment with lr
    # Return final accuracy
    pass

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
```

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional model architectures (ResNet, MobileNet)
- More datasets (CIFAR-10, CIFAR-100)
- Alternative aggregation strategies (FedProx, FedOpt, FedAdam)
- Enhanced visualization tools
- Real-time monitoring dashboard
- Differential privacy integration

---

## 📜 License

Apache-2.0 License

---

## 🙏 Acknowledgments

- **Flower Team**: For the excellent federated learning framework
- **PyTorch Team**: For the deep learning foundation
- **Fashion-MNIST**: Dataset by Zalando Research

---

## 📧 Contact & Support

- **Flower Documentation**: https://flower.ai/docs
- **GitHub Issues**: For bug reports and feature requests
- **Flower Community**: https://flower.ai/join-slack

---

**⭐ Star this repository if useful for your federated learning research or projects!**
