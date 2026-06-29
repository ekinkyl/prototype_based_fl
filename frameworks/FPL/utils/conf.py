import os
import random
from pathlib import Path

import numpy as np
import torch

# Expected layout (same on every machine):
#   pipeline/
#   ├── shared_datasets/
#   └── frameworks/FPL/utils/conf.py  <-- this file
_PIPELINE_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_ROOT = _PIPELINE_ROOT / "shared_datasets"


def get_device(device_id) -> torch.device:
    return torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")


def data_path() -> str:
    """
    Central dataset directory for the benchmark pipeline.
    Override with FL_BENCHMARK_DATA_ROOT when running via main_runner.py or on a cluster.
    """
    root = os.environ.get("FL_BENCHMARK_DATA_ROOT", str(_DEFAULT_DATA_ROOT))
    path = Path(root).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path) + os.sep


def base_path() -> str:
    return './data/'


def checkpoint_path() -> str:
    return './checkpoint/'


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
