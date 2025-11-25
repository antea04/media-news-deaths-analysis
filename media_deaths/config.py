from pathlib import Path
import yaml
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration loader for media deaths analysis."""

    def __init__(self, config_path: str | Path):
        """Load configuration from YAML file.

        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path, "r") as f:
            self._config = yaml.safe_load(f)

        self._load_config()

    def _load_config(self):
        """Load all configuration values from the config dict."""
        # Colors used for charts
        self.FIXED_COLORS = {
            k: v["color"] for k, v in self._config["causes_death"].items()
        }
        # Queries for each cause of death
        self.QUERIES = {k: v["query"] for k, v in self._config["causes_death"].items()}
        # All causes of death
        self.CAUSES_OF_DEATH_ALL = list(self._config["causes_death"].keys())

        # Media Cloud Token
        self.MC_API_TOKEN = os.getenv("MC_API_TOKEN")

        # Set overall variables for analysis
        self.YEAR = self._config["config"]["year"]
        self.VERBOSE = self._config["config"]["verbose"]
        self.LANGUAGE = self._config["config"]["language"]
        self.RERUN_QUERIES = self._config["config"]["rerun_queries"]
        self.RUN_SINGLE_QUERIES = self._config["config"]["run_single_queries"]
        self.OVERWRITE = self._config["config"]["overwrite"]
        self.USE_SAVED_RESULTS = self._config["config"]["use_saved_results"]
        self.API_SLEEP = self._config["config"]["api_sleep"]

        # Media outlets information
        self.OUTLETS = self._config["outlets"]
        # Media collections information
        self.COLLECTIONS = self._config["collections"]

    def check_valid_causes(self, causes: list[str]) -> None:
        """Check if the provided causes are valid.

        Args:
            causes: List of cause of death identifiers to validate

        Raises:
            ValueError: If any causes are not in the configuration
        """
        invalid_causes = [c for c in causes if c not in self.CAUSES_OF_DEATH_ALL]
        if invalid_causes:
            raise ValueError(
                f"Invalid causes of death: {invalid_causes}. "
                f"Valid causes are: {self.CAUSES_OF_DEATH_ALL}"
            )
