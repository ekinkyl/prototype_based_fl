from abc import ABC, abstractmethod
import os
from pathlib import Path

class FrameworkAdapter(ABC):
    """Base adapter that each framework-specific adapter must implement."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the framework name (e.g., 'FedProto')."""
        pass
    
    @abstractmethod
    def get_repo_path(self) -> Path:
        """Return the path where this framework's code is cloned."""
        pass
    
    @abstractmethod
    def get_entry_point(self) -> str:
        """Return the main script to execute (e.g., 'main.py' or 'exps/federated_main.py')."""
        pass
    
    @abstractmethod
    def build_cli_args(self, global_config: dict, overrides: dict) -> list[str]:
        """Map global config and overrides into a list of CLI arguments."""
        pass
    
    @abstractmethod
    def setup_data_path(self, shared_data_path: Path) -> None:
        """
        Configure the framework to use the shared datasets.
        This may involve creating symlinks or updating config files.
        """
        pass
    
    def get_env_vars(self) -> dict:
        """Optional environment variables to inject before running."""
        return os.environ.copy()
        
    def get_cwd(self) -> Path:
        """Return the working directory for subprocess execution. Usually the repo root."""
        return self.get_repo_path()
        
    def validate(self) -> bool:
        """Check if the repo exists and is ready to run."""
        repo_path = self.get_repo_path()
        return repo_path.exists() and repo_path.is_dir()
