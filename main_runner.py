#!/usr/bin/env python3
"""
Central orchestrator for benchmarking multiple Federated Learning frameworks
under identical conditions without modifying their source code.

Usage:
    python main_runner.py                          # run all active frameworks
    python main_runner.py --config central_config.yaml
    python main_runner.py --framework FedProto     # run a single framework
    python main_runner.py --list-frameworks
    python main_runner.py --dry-run
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required. Install with: pip install pyyaml"
    ) from exc


# =============================================================================
# WORKSPACE LAYOUT (clone official repos here)
# =============================================================================
# pipeline/
# ├── central_config.yaml
# ├── main_runner.py
# ├── shared_datasets/          <- CIFAR-10, MNIST, etc.
# └── frameworks/
#     ├── FedAvg/               <- git clone <FedAvg-repo-url>
#     ├── FedProto/             <- git clone <FedProto-repo-url>
#     ├── FedNH/
#     ├── FPL/
#     ├── FedPLVM/
#     ├── FedTGP/
#     ├── FedPall/
#     ├── FedDAP/
#     ├── FedGMKD/
#     └── FedPCL/
# =============================================================================

WORKSPACE_ROOT = Path(__file__).resolve().parent
FRAMEWORKS_DIR = WORKSPACE_ROOT / "frameworks"
SHARED_DATASETS_DIR = WORKSPACE_ROOT / "shared_datasets"
DEFAULT_CONFIG = WORKSPACE_ROOT / "central_config.yaml"


# =============================================================================
# FRAMEWORK ADAPTER REGISTRY
# Add a new entry here after cloning a repo into frameworks/<Name>/.
#
# Each adapter describes:
#   - entry_script: relative path inside the cloned repo
#   - param_map: central_config dotted keys -> CLI flag templates
#   - env_map: central_config dotted keys -> environment variable names
#   - static_args: always-appended CLI flags
#   - data_path: how to inject shared_datasets/ without editing repo code
# =============================================================================

@dataclass(frozen=True)
class DataPathStrategy:
    """How to redirect a framework to shared_datasets/."""

    cli_flag: str | None = None          # e.g. "--datadir"
    env_var: str | None = None           # e.g. "DATA_DIR"
    symlink_into_repo: str | None = None # e.g. "./data" -> symlink to shared root
    append_dataset_name: bool = True     # pass data.root/name vs root only


@dataclass(frozen=True)
class FrameworkAdapter:
    name: str
    repo_dir: str                        # subfolder under frameworks/
    entry_script: str                    # main.py, run.py, train.py, etc.
    # param_map values:
    #   str with {value} placeholder -> "--flag {value}"
    #   dict with keys: flag, transform (optional), boolean (optional)
    param_map: Mapping[str, str | dict[str, Any]]
    env_map: Mapping[str, str] = field(default_factory=dict)
    static_args: Sequence[str] = field(default_factory=tuple)
    data_path: DataPathStrategy = field(
        default_factory=lambda: DataPathStrategy(
            cli_flag="--datadir",
            env_var="DATA_DIR",
            symlink_into_repo="./data",
        )
    )
    python: str = sys.executable
    extra_env: Mapping[str, str] = field(default_factory=dict)


def _flag(name: str, *, boolean: bool = False, transform: str | None = None) -> dict[str, Any]:
    return {"flag": name, "boolean": boolean, "transform": transform}


FRAMEWORK_ADAPTERS: dict[str, FrameworkAdapter] = {
    # -------------------------------------------------------------------------
    # FedAvg — classic baseline; most repos expose similar argparse flags.
    # Tune param_map after inspecting: frameworks/FedAvg/main.py --help
    # -------------------------------------------------------------------------
    "FedAvg": FrameworkAdapter(
        name="FedAvg",
        repo_dir="FedAvg",
        entry_script="main.py",
        param_map={
            "data.name": _flag("--dataset"),
            "data.partition": _flag("--partition"),
            "data.num_clients": _flag("--num_clients", transform="int"),
            "training.global_rounds": _flag("--rounds", transform="int"),
            "training.local_epochs": _flag("--local_epochs", transform="int"),
            "training.batch_size": _flag("--batch_size", transform="int"),
            "training.learning_rate": _flag("--lr", transform="float"),
            "model.backbone": _flag("--model"),
            "experiment.seed": _flag("--seed", transform="int"),
            "runtime.device": _flag("--device"),
        },
        env_map={
            "data.root": "DATA_ROOT",
            "data.name": "DATASET_NAME",
        },
        static_args=("--alg", "fedavg"),
        data_path=DataPathStrategy(
            cli_flag="--datadir",
            env_var="DATA_DIR",
            symlink_into_repo="./data",
        ),
    ),
    "FedTGP": FrameworkAdapter(
        name="FedTGP",
        repo_dir="FedTGP",
        entry_script="system/main.py",
        param_map={
            "data.num_clients": _flag("-nc", transform="int"),
            "training.global_rounds": _flag("-gr", transform="int"),
            "training.local_epochs": _flag("-ls", transform="int"),
            "training.batch_size": _flag("-lbs", transform="int"),
            "training.learning_rate": _flag("-lr", transform="float"),
        },
        static_args=("-data", "mnist", "-m", "HtFE3_MNIST", "-algo", "FedTGP", "-go", "test"),
        data_path=DataPathStrategy(
            env_var="DATA_ROOT",
        ),
    ),
    "FedProto": FrameworkAdapter(
        name="FedProto",
        repo_dir="FedProto",
        entry_script="main.py",
        param_map={
            "data.name": _flag("--dataset"),
            "data.num_clients": _flag("--num_users", transform="int"),
            "training.global_rounds": _flag("--epochs", transform="int"),
            "training.local_epochs": _flag("--local_ep", transform="int"),
            "training.batch_size": _flag("--batch_size", transform="int"),
            "training.learning_rate": _flag("--lr", transform="float"),
            "model.backbone": _flag("--model"),
            "experiment.seed": _flag("--seed", transform="int"),
            "runtime.gpu_id": _flag("--gpu", transform="int"),
        },
        env_map={"data.root": "DATA_PATH"},
        static_args=("--alg", "fedproto"),
        data_path=DataPathStrategy(
            cli_flag="--data",
            env_var="DATA_PATH",
            symlink_into_repo="./dataset",
        ),
    ),
    "FedNH": FrameworkAdapter(
        name="FedNH",
        repo_dir="FedNH",
        entry_script="main.py",
        param_map={
            # --num_clients: number of federated clients
            "data.num_clients":       _flag("--num_clients", transform="int"),
            # --num_rounds: global communication rounds
            "training.global_rounds": _flag("--num_rounds", transform="int"),
            # --num_epochs: local training epochs per round
            "training.local_epochs":  _flag("--num_epochs", transform="int"),
            # --client_lr: client-side learning rate
            "training.learning_rate": _flag("--client_lr", transform="float"),
            # --global_seed: random seed
            "experiment.seed":        _flag("--global_seed", transform="int"),
        },
        # FedNH-specific static args:
        #   --yamlfile: the internal base config (model, batch_size, etc.)
        #   --strategy: selects FedNH algorithm
        #   --partition + --beta: data heterogeneity via Dirichlet
        #   --participate_ratio 1.0: all clients train every round
        #   --no_norm True: disables batch/group norm (paper default)
        #   --use_wandb False: no wandb logging
        #   --device: set at runtime via config
        static_args=(
            "--yamlfile", "base_config.yaml",
            "--strategy", "FedNH",
            "--partition", "noniid-label-distribution",
            "--beta", "0.5",
            "--participate_ratio", "1.0",
            "--no_norm", "True",
            "--use_wandb", "False",
            "--device", "cpu",
        ),
        data_path=DataPathStrategy(
            cli_flag=None,                 # FedNH reads data via get_datasets()
            env_var="FL_BENCHMARK_DATA_ROOT",
            symlink_into_repo=None,
            append_dataset_name=False,
        ),
    ),
    "FPL": FrameworkAdapter(
        name="FPL",
        repo_dir="FPL",
        entry_script="main.py",
        param_map={
            # --dataset: must be "fl_digits" or "fl_officecaltech"
            "data.name":              _flag("--dataset"),
            # --parti_num: number of federated participants (clients)
            "data.num_clients":       _flag("--parti_num", transform="int"),
            # --communication_epoch: global communication rounds
            "training.global_rounds": _flag("--communication_epoch", transform="int"),
            # --local_epoch: local training epochs per round
            "training.local_epochs":  _flag("--local_epoch", transform="int"),
            # --seed
            "experiment.seed":        _flag("--seed", transform="int"),
            # --device_id: GPU device index (0 for CPU fallback)
            "runtime.gpu_id":         _flag("--device_id", transform="int"),
        },
        # NOTE: --model selects the FL *algorithm* in FPL (fpl/fedavg/moon),
        #       NOT the backbone. Backbone is chosen automatically per dataset.
        #       --lr and --batch_size are overridden by FPL's best_args dict.
        static_args=("--model", "fpl"),
        data_path=DataPathStrategy(
            cli_flag=None,                 # FPL reads data_path() from env var
            env_var="FL_BENCHMARK_DATA_ROOT",
            symlink_into_repo=None,
            append_dataset_name=False,     # FPL manages subdirs internally
        ),
    ),
    "FedPLVM": FrameworkAdapter(
        name="FedPLVM",
        repo_dir="FedPLVM",
        entry_script="main.py",
        param_map={
            "data.num_clients": _flag("--num_clients", transform="int"),
            "training.global_rounds": _flag("--rounds", transform="int"),
            "training.local_epochs": _flag("--train_ep", transform="int"),
            "training.batch_size": _flag("--local_bs", transform="int"),
            "training.learning_rate": _flag("--lr", transform="float"),
            "experiment.seed": _flag("--seed", transform="int"),
        },
        static_args=(
            "--dataset", "digit",
            "--model", "resnet",
            "--label_iid", "False",
            "--test_bs", "64",
        ),
        data_path=DataPathStrategy(
            cli_flag=None,
            env_var="DATA_ROOT",
            symlink_into_repo=None,
            append_dataset_name=False
        ),
    ),

    "FedPall": FrameworkAdapter(
        name="FedPall",
        repo_dir="FedPall",
        entry_script="run.py",
        param_map={
            "data.name": _flag("--dataset"),
            "training.global_rounds": _flag("--rounds", transform="int"),
            "training.local_epochs": _flag("--local_epochs", transform="int"),
            "training.learning_rate": _flag("--lr", transform="float"),
            "experiment.seed": _flag("--seed", transform="int"),
        },
        data_path=DataPathStrategy(cli_flag="--data_dir", env_var="DATA_PATH"),
    ),
    "FedDAP": FrameworkAdapter(
        name="FedDAP",
        repo_dir="FedDAP",
        entry_script="main.py",
        param_map={
            "data.name": _flag("--dataset"),
            "data.num_clients": _flag("--num_clients", transform="int"),
            "training.global_rounds": _flag("--rounds", transform="int"),
            "training.local_epochs": _flag("--local_epochs", transform="int"),
            "training.learning_rate": _flag("--lr", transform="float"),
            "experiment.seed": _flag("--seed", transform="int"),
        },
        data_path=DataPathStrategy(cli_flag="--datadir", env_var="DATA_DIR"),
    ),
    "FedGMKD": FrameworkAdapter(
        name="FedGMKD",
        repo_dir="FedGMKD",
        entry_script="main.py",
        param_map={
            "data.name": _flag("--dataset"),
            "data.num_clients": _flag("--num_users", transform="int"),
            "training.global_rounds": _flag("--epochs", transform="int"),
            "training.local_epochs": _flag("--local_ep", transform="int"),
            "training.learning_rate": _flag("--lr", transform="float"),
            "experiment.seed": _flag("--seed", transform="int"),
        },
        data_path=DataPathStrategy(cli_flag="--data", env_var="DATA_PATH"),
    ),
    "FedPCL": FrameworkAdapter(
        name="FedPCL",
        repo_dir="FedPCL",
        entry_script="run.py",
        param_map={
            "data.name": _flag("--dataset"),
            "data.num_clients": _flag("--num_clients", transform="int"),
            "training.global_rounds": _flag("--rounds", transform="int"),
            "training.local_epochs": _flag("--local_epochs", transform="int"),
            "training.batch_size": _flag("--batch_size", transform="int"),
            "training.learning_rate": _flag("--lr", transform="float"),
            "experiment.seed": _flag("--seed", transform="int"),
        },
        data_path=DataPathStrategy(cli_flag="--data_root", env_var="DATA_ROOT"),
    ),
}


# =============================================================================
# Config utilities
# =============================================================================

TransformFn = Callable[[Any], str]


TRANSFORMS: dict[str, TransformFn] = {
    "int": lambda v: str(int(v)),
    "float": lambda v: str(float(v)),
    "str": lambda v: str(v),
    "bool": lambda v: "true" if v else "false",
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def get_by_dotted_key(config: Mapping[str, Any], dotted_key: str) -> Any:
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(f"Config key not found: {dotted_key}")
        node = node[part]
    return node


def resolve_data_root(config: Mapping[str, Any]) -> Path:
    root = Path(config["data"]["root"])
    if not root.is_absolute():
        root = (WORKSPACE_ROOT / root).resolve()
    return root


def resolve_dataset_path(config: Mapping[str, Any], adapter: FrameworkAdapter) -> Path:
    root = resolve_data_root(config)
    if adapter.data_path.append_dataset_name:
        return root / str(config["data"]["name"])
    return root


def apply_framework_overrides(config: dict[str, Any], framework_name: str) -> dict[str, Any]:
    overrides = config.get("overrides", {}).get(framework_name, {})
    if not overrides:
        return config
    return deep_merge(config, overrides)


def build_cli_args(config: Mapping[str, Any], adapter: FrameworkAdapter) -> list[str]:
    args: list[str] = list(adapter.static_args)
    for dotted_key, spec in adapter.param_map.items():
        try:
            value = get_by_dotted_key(config, dotted_key)
        except KeyError:
            continue

        if isinstance(spec, str):
            if "{value}" in spec:
                args.append(spec.format(value=value))
            else:
                args.extend([spec, str(value)])
            continue

        flag = spec["flag"]
        if spec.get("boolean"):
            if bool(value):
                args.append(flag)
            continue

        transform_name = spec.get("transform")
        if transform_name:
            value = TRANSFORMS[transform_name](value)
        else:
            value = str(value)
        args.extend([flag, value])

    data_path = resolve_dataset_path(config, adapter)
    if adapter.data_path.cli_flag:
        args.extend([adapter.data_path.cli_flag, str(data_path)])

    return args


def build_env(config: Mapping[str, Any], adapter: FrameworkAdapter) -> dict[str, str]:
    env = os.environ.copy()
    env.update(adapter.extra_env)

    data_root = resolve_data_root(config)
    dataset_path = resolve_dataset_path(config, adapter)

    if adapter.data_path.env_var:
        env[adapter.data_path.env_var] = str(dataset_path)

    for dotted_key, env_name in adapter.env_map.items():
        try:
            value = get_by_dotted_key(config, dotted_key)
        except KeyError:
            continue
        if dotted_key == "data.root":
            env[env_name] = str(data_root)
        elif dotted_key == "data.name":
            env[env_name] = str(value)
        else:
            env[env_name] = str(value)

    # Standard cross-repo variables (repos that honor them need zero changes).
    env.setdefault("FL_BENCHMARK_DATA_ROOT", str(data_root))
    env.setdefault("FL_BENCHMARK_DATASET", str(config["data"]["name"]))
    env.setdefault("FL_BENCHMARK_SEED", str(config["experiment"]["seed"]))
    env.setdefault("FL_BENCHMARK_OUTPUT_DIR", str(config["experiment"]["output_dir"]))

    runtime = config.get("runtime", {})
    gpu_id = runtime.get("gpu_id")
    if runtime.get("device") == "cuda" and gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    elif runtime.get("device") == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    return env


# =============================================================================
# Per-framework requirements management
# =============================================================================

def find_requirements_file(adapter: FrameworkAdapter) -> Path | None:
    """Return the path to a framework's requirements.txt, or None."""
    req_path = FRAMEWORKS_DIR / adapter.repo_dir / "requirements.txt"
    return req_path if req_path.is_file() else None


def install_requirements(adapter: FrameworkAdapter) -> int:
    """pip install -r the framework's requirements.txt. Returns exit code."""
    req_path = find_requirements_file(adapter)
    if req_path is None:
        logging.info("[%s] No requirements.txt found — skipping.", adapter.name)
        return 0
    logging.info("[%s] Installing dependencies from %s ...", adapter.name, req_path)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_path)],
        check=False,
    )
    if result.returncode != 0:
        logging.error("[%s] pip install failed (exit code %d).", adapter.name, result.returncode)
    else:
        logging.info("[%s] Dependencies installed successfully.", adapter.name)
    return result.returncode


def warn_if_requirements_exist(adapter: FrameworkAdapter) -> None:
    """Log a warning if a framework has a requirements.txt that hasn't been
    explicitly installed via --install-deps."""
    req_path = find_requirements_file(adapter)
    if req_path is not None:
        logging.warning(
            "[%s] requirements.txt found at %s. "
            "Run with --install-deps to auto-install, or install manually:\n"
            "    pip install -r %s",
            adapter.name, req_path, req_path,
        )


def ensure_data_symlink(repo_path: Path, adapter: FrameworkAdapter, dataset_path: Path) -> None:
    """Optional fallback when a repo hardcodes a relative ./data folder."""
    if not adapter.data_path.symlink_into_repo:
        return

    link_path = (repo_path / adapter.data_path.symlink_into_repo).resolve()
    dataset_path.mkdir(parents=True, exist_ok=True)

    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and link_path.resolve() == dataset_path.resolve():
            return
        if link_path.is_dir() and not link_path.is_symlink():
            return  # real directory already present; do not destroy user data
        link_path.unlink(missing_ok=True)

    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(dataset_path, target_is_directory=True)


def build_command(
    config: Mapping[str, Any],
    adapter: FrameworkAdapter,
    *,
    require_repo: bool = True,
) -> tuple[list[str], dict[str, str], Path]:
    repo_path = (FRAMEWORKS_DIR / adapter.repo_dir).resolve()
    entry = repo_path / adapter.entry_script
    if require_repo and not entry.exists():
        raise FileNotFoundError(
            f"[{adapter.name}] Entry script not found: {entry}\n"
            f"Clone the official repository into: {repo_path}"
        )

    dataset_path = resolve_dataset_path(config, adapter)
    if require_repo:
        ensure_data_symlink(repo_path, adapter, dataset_path)
        
        # --- FedTGP Special Hook ---
        # Run original data generation if dataset/mnist/train is missing
        if adapter.name == "FedTGP":
            gen_script = repo_path / "dataset" / "generate_mnist.py"
            data_dir = repo_path / "dataset" / "mnist"
            if gen_script.exists() and not (data_dir / "train").exists():
                logging.info("[%s] Running data generation script...", adapter.name)
                subprocess.run(
                    [sys.executable, str(gen_script)], 
                    cwd=str(repo_path / "dataset"), 
                    check=True
                )


    cli_args = build_cli_args(config, adapter)
    env = build_env(config, adapter)
    command = [adapter.python, str(entry), *cli_args]
    return command, env, repo_path


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def run_subprocess(
    command: Sequence[str],
    env: Mapping[str, str],
    cwd: Path,
    *,
    dry_run: bool,
    timeout_seconds: int,
    log_file: Path,
) -> int:
    logging.info("CWD: %s", cwd)
    logging.info("CMD: %s", shlex.join(command))
    logging.info("ENV (benchmark): DATA_ROOT=%s", env.get("FL_BENCHMARK_DATA_ROOT"))

    if dry_run:
        logging.info("Dry-run enabled; skipping execution.")
        return 0

    with log_file.open("a", encoding="utf-8") as fp:
        fp.write(f"\n{'=' * 80}\n")
        fp.write(f"CWD: {cwd}\n")
        fp.write(f"CMD: {shlex.join(command)}\n")
        fp.write(f"{'=' * 80}\n")

    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=dict(env),
            check=False,
            timeout=timeout_seconds or None,
        )
        return int(completed.returncode)
    except subprocess.TimeoutExpired:
        logging.error("Process timed out after %s seconds.", timeout_seconds)
        return 124


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {path}")
    return config


def validate_config(config: Mapping[str, Any], selected: Sequence[str]) -> None:
    missing = [name for name in selected if name not in FRAMEWORK_ADAPTERS]
    if missing:
        known = ", ".join(sorted(FRAMEWORK_ADAPTERS))
        raise ValueError(
            f"Unknown framework(s): {missing}. Registered adapters: {known}"
        )


def run_benchmark(
    config_path: Path,
    *,
    framework_filter: str | None,
    dry_run: bool | None,
    install_deps: bool = False,
) -> int:
    config = load_config(config_path)
    experiment = config.setdefault("experiment", {})
    runtime = config.setdefault("runtime", {})

    active = config.get("frameworks", {}).get("active", [])
    if framework_filter:
        selected = [framework_filter]
    else:
        selected = list(active)

    if not selected:
        raise ValueError("No frameworks selected. Set frameworks.active or use --framework.")

    validate_config(config, selected)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(experiment.get("output_dir", "./results"))
    if not output_dir.is_absolute():
        output_dir = (WORKSPACE_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    orchestrator_log = output_dir / f"orchestrator_{timestamp}.log"
    setup_logging(orchestrator_log)

    SHARED_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    FRAMEWORKS_DIR.mkdir(parents=True, exist_ok=True)

    effective_dry_run = dry_run if dry_run is not None else bool(experiment.get("dry_run", False))
    timeout_seconds = int(runtime.get("timeout_seconds", 0))

    logging.info("Experiment: %s", experiment.get("name", "unnamed"))
    logging.info("Config: %s", config_path)
    logging.info("Frameworks: %s", ", ".join(selected))

    exit_codes: dict[str, int] = {}

    for name in selected:
        adapter = FRAMEWORK_ADAPTERS[name]

        # --- Per-framework dependency management ---
        if install_deps:
            dep_code = install_requirements(adapter)
            if dep_code != 0:
                logging.error("[%s] Dependency installation failed. Aborting.", name)
                exit_codes[name] = dep_code
                continue
        else:
            warn_if_requirements_exist(adapter)

        framework_config = apply_framework_overrides(config, name)
        command, env, cwd = build_command(
            framework_config,
            adapter,
            require_repo=not effective_dry_run,
        )

        framework_log = output_dir / f"{name}_{timestamp}.log"
        logging.info("--- Running %s ---", name)
        code = run_subprocess(
            command,
            env,
            cwd,
            dry_run=effective_dry_run,
            timeout_seconds=timeout_seconds,
            log_file=framework_log,
        )
        exit_codes[name] = code
        if code != 0:
            logging.error("%s finished with exit code %d", name, code)
        else:
            logging.info("%s completed successfully.", name)

    failed = [name for name, code in exit_codes.items() if code != 0]
    if failed:
        logging.error("Failed frameworks: %s", ", ".join(failed))
        return 1
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified FL benchmark orchestrator")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to central_config.yaml",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default=None,
        help="Run only this framework (overrides frameworks.active)",
    )
    parser.add_argument(
        "--list-frameworks",
        action="store_true",
        help="List registered framework adapters and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print subprocess commands without executing",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Auto-install each framework's requirements.txt before running",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_frameworks:
        for name, adapter in sorted(FRAMEWORK_ADAPTERS.items()):
            repo = FRAMEWORKS_DIR / adapter.repo_dir
            req = find_requirements_file(adapter)
            req_status = f"  deps={req}" if req else "  deps=none"
            print(f"{name:10}  repo={repo}  entry={adapter.entry_script}{req_status}")
        return 0

    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")

    return run_benchmark(
        args.config,
        framework_filter=args.framework,
        dry_run=True if args.dry_run else None,
        install_deps=args.install_deps,
    )


if __name__ == "__main__":
    raise SystemExit(main())
