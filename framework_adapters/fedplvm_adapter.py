from pathlib import Path
import os
import subprocess
from .base_adapter import FrameworkAdapter

class FedPLVMAdapter(FrameworkAdapter):
    def __init__(self, workspace_root: Path):
        self.repo_path = workspace_root / "frameworks" / "FedPLVM"
        
    def get_name(self) -> str:
        return "FedPLVM"
        
    def get_repo_path(self) -> Path:
        return self.repo_path
        
    def get_entry_point(self) -> str:
        return "main.py"
        
    def setup_data_path(self, shared_data_path: Path) -> None:
        target_data_dir = self.repo_path / "data"
        if not target_data_dir.exists():
            try:
                if os.name == 'nt':
                    subprocess.run(["cmd", "/c", "mklink", "/J", str(target_data_dir), str(shared_data_path)], check=True, capture_output=True)
                else:
                    os.symlink(shared_data_path, target_data_dir)
            except Exception as e:
                print(f"Warning: Failed to create symlink for {self.get_name()}: {e}")

    def build_cli_args(self, global_config: dict, overrides: dict) -> list[str]:
        args = [
            "--dataset", global_config.get("dataset", "mnist"),
            "--num_users", str(global_config.get("num_clients", 20)),
            "--epochs", str(global_config.get("communication_rounds", 100)),
            "--local_ep", str(global_config.get("local_epochs", 5)),
            "--lr", str(global_config.get("learning_rate", 0.01)),
            "--local_bs", str(global_config.get("batch_size", 64)),
            "--seed", str(global_config.get("seed", 42)),
        ]
        
        for k, v in overrides.items():
            args.extend([f"--{k}", str(v)])
            
        return args
