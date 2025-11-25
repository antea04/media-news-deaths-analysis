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

        # Set data variables (language, year, data loader)
        data_config = self._config.get("data", {})
        self.YEAR = data_config.get("year")
        if not self.YEAR:
            raise ValueError("Missing required field 'data.year' in config")
        self.LANGUAGE = data_config.get("language")
        if not self.LANGUAGE:
            raise ValueError("Missing required field 'data.language' in config")
        self.DATA_LOADER = data_config.get("data_loader")
        if not self.DATA_LOADER:
            raise ValueError(
                "Missing required field 'data.data_loader'. "
                "Please specify which data loader to use (e.g., 'catalan_gencat')"
            )

        # Set runtime variables (with defaults)
        runtime_config = self._config.get("runtime", {})
        self.VERBOSE = runtime_config.get("verbose", True)
        self.RERUN_QUERIES = runtime_config.get("rerun_queries", True)
        self.RUN_SINGLE_QUERIES = runtime_config.get("run_single_queries", False)
        self.OVERWRITE = runtime_config.get("overwrite", False)
        self.USE_SAVED_RESULTS = runtime_config.get("use_saved_results", False)
        self.API_SLEEP = runtime_config.get("api_sleep", 10)

        # Validate that the loader exists
        self._validate_data_loader()

        # Media outlets information
        self.OUTLETS = self._config["outlets"]
        # Media collections information
        self.COLLECTIONS = self._config["collections"]

    def _validate_data_loader(self):
        """Validate that the configured data loader exists.

        Raises:
            ValueError: If the data loader is not registered
        """
        from media_deaths.data_loaders import list_loaders, discover_loaders

        # Ensure loaders are discovered before validation
        discover_loaders()

        available_loaders = list_loaders()
        if self.DATA_LOADER not in available_loaders:
            raise ValueError(
                f"Unknown data loader: '{self.DATA_LOADER}'. "
                f"Available loaders: {', '.join(available_loaders) if available_loaders else '(none)'}"
            )

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
