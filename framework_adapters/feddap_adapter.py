from pathlib import Path
import os
import subprocess
from .base_adapter import FrameworkAdapter

class FedDAPAdapter(FrameworkAdapter):
    def __init__(self, workspace_root: Path):
        self.repo_path = workspace_root / "frameworks" / "FedDAP"
        
    def get_name(self) -> str:
        return "FedDAP"
        
    def get_repo_path(self) -> Path:
        return self.repo_path
        
    def get_entry_point(self) -> str:
        return "main.py"
        
    def setup_data_path(self, shared_data_path: Path) -> None:
        target_data_dir = self.repo_path / "datasets" / "pic_cls"
        if not target_data_dir.parent.exists():
            target_data_dir.parent.mkdir(parents=True, exist_ok=True)
            
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
            "--dataset", global_config.get("dataset", "fl_digits"),
            "--model", "feddap",
            "--communication_epoch", str(global_config.get("communication_rounds", 100)),
            "--local_epoch", str(global_config.get("local_epochs", 10)),
            "--parti_num", str(global_config.get("num_clients", 12)),
            "--device_id", str(global_config.get("gpu_id", 0)),
            "--seed", str(global_config.get("seed", 0)),
        ]
        
        for k, v in overrides.items():
            args.extend([f"--{k}", str(v)])
            
        return args
