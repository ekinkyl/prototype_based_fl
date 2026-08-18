from pathlib import Path
import os
from .base_adapter import FrameworkAdapter

class FedPCLAdapter(FrameworkAdapter):
    def __init__(self, workspace_root: Path):
        self.repo_path = workspace_root / "frameworks" / "FedPCL"
        
    def get_name(self) -> str:
        return "FedPCL"
        
    def get_repo_path(self) -> Path:
        return self.repo_path
        
    def get_entry_point(self) -> str:
        return "exps/federated_main.py"
        
    def setup_data_path(self, shared_data_path: Path) -> None:
        # FedPCL has a --data_dir CLI flag, no symlink strictly needed,
        # but we can store the path to use in build_cli_args
        self.shared_data_path = shared_data_path

    def build_cli_args(self, global_config: dict, overrides: dict) -> list[str]:
        args = [
            "--dataset", global_config.get("dataset", "digit"),
            "--alg", "fedpcl",
            "--num_users", str(global_config.get("num_clients", 20)),
            "--rounds", str(global_config.get("communication_rounds", 100)),
            "--lr", str(global_config.get("learning_rate", 0.001)),
            "--local_bs", str(global_config.get("batch_size", 32)),
            "--model", global_config.get("model_backbone", "cnn"),
            "--alpha", str(global_config.get("alpha", 1)),
            "--data_dir", str(getattr(self, 'shared_data_path', './data/')),
        ]
        
        for k, v in overrides.items():
            args.extend([f"--{k}", str(v)])
            
        return args
