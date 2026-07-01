# Federated Learning Benchmark Pipeline

A standardized evaluation pipeline designed to benchmark multiple existing, standalone Prototype-based Federated Learning repositories under identical conditions. Instead of rewriting complex baseline code, this pipeline acts as an orchestrator that seamlessly wraps official repositories, injecting central configurations via CLI arguments and shared environments.

## Architecture

The workspace is organized into a modular structure:

```
pipeline/
├── central_config.yaml     # Global hyperparameters, dataset paths, and framework selectors
├── main_runner.py          # Central orchestrator that parses config and triggers specific repos
├── README.md               # This file
├── shared_datasets/        # Centralized data directory (MNIST) shared across all repos
└── frameworks/             # Cloned official repositories
    └── FPL/                # First integrated framework (Federated Prototypes Learning)
```

## How It Works

1. **Centralized Configuration:** The `central_config.yaml` is the single source of truth for the experiment.
2. **Adapter Pattern:** The `main_runner.py` contains a `FrameworkAdapter` for each integrated repo. It maps the global YAML settings into the exact CLI arguments and environment variables expected by the specific repository.
3. **Data Unification:** The orchestrator enforces that all wrapped repositories read from the unified `shared_datasets/` folder by overriding their internal data pathing (using environment variables like `FL_BENCHMARK_DATA_ROOT`).
4. **Dependency Management:** Each framework maintains its own `requirements.txt`. The orchestrator can auto-install these before execution.

## Current Configuration Overview

We are currently benchmarking **FPL (Federated Prototypes Learning)** on an initial baseline test.

### Experiment Settings
- **Dataset:** MNIST (configured via FPL's `fl_digits` scenario, but patched to strictly use only the MNIST domain).
- **Data Partition:** Domain Skew (FPL's default architecture approach). //for now 
- **Total Clients:** 10 clients.
- **Hardware:** CPU (Configured for local testing. Switch `device: "cuda"` and `gpu_id: 0` for A100 Colab runs).

### Model & Federated Learning Hyperparameters
- **Frameworks:** FPL, FedNH, FedPLVM, FedTGP, FedPall, FedDAP, FedGMKD, FedPCL, FedProto, FedAvg.
- **Model Backbone:** ResNet10.
- **Global Communication Rounds:** 20.
- **Local Epochs (per client):** 10.
- **Batch Size:** 64.
- **Learning Rate:** 0.01 (using SGD with 0.9 momentum and weight decay).
- **Seed:** 42 (ensuring deterministic behavior across benchmarks).

## Usage

### 1. View Available Frameworks
List all registered framework adapters and their dependency status:
```bash
python main_runner.py --list-frameworks
```

### 2. Install Dependencies & Run
You can automatically install a specific framework's requirements and run it in one command:
```bash
python main_runner.py --framework FPL --install-deps
```

### 3. Dry-Run Verification
Check the exact command the orchestrator will trigger without actually running it:
```bash
python main_runner.py --framework FPL --dry-run
```

## Adding a New Framework

To integrate a new repository (e.g., FedProto):
1. Clone the official repository into `frameworks/FedProto`.
2. Extract its dependencies into `frameworks/FedProto/requirements.txt`.
3. Open `main_runner.py` and register a new `FrameworkAdapter` under the `FRAMEWORK_ADAPTERS` dictionary.
4. Map the `central_config.yaml` keys to the new repo's specific CLI flags.
