from pathlib import Path
import os
import subprocess
from .base_adapter import FrameworkAdapter

class FedPallAdapter(FrameworkAdapter):
    def __init__(self, workspace_root: Path):
        self.repo_path = workspace_root / "frameworks" / "FedPall"
        
    def get_name(self) -> str:
        return "FedPall"
        
    def get_repo_path(self) -> Path:
        return self.repo_path
        
    def get_entry_point(self) -> str:
        return "exps/federated_main.py"
        
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
            "--iters", str(global_config.get("communication_rounds", 100)),
            "--batch", str(global_config.get("batch_size", 64)),
        ]
        
        for k, v in overrides.items():
            args.extend([f"--{k}", str(v)])
            
        return args
