"""Data loader registry for media deaths analysis.

This module provides a registry pattern for data loaders. Each loader is responsible
for fetching and formatting mortality data from a specific source.

Each loader must:
1. Be decorated with @register_loader("loader_name")
2. Accept a Config object as parameter
3. Return a pandas DataFrame with columns: ['code', 'cause', 'deaths', 'year']
"""

from typing import Callable
import pandas as pd

# Import Config at runtime to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from media_deaths.config import Config

# Registry of available data loaders
DATA_LOADERS: dict[str, Callable[["Config"], pd.DataFrame]] = {}


def register_loader(name: str):
    """Decorator to register a data loader function.

    Args:
        name: Unique identifier for the loader (referenced in config files)

    Example:
        @register_loader("catalan_gencat")
        def load_data(config: Config) -> pd.DataFrame:
            # Implementation here
            pass
    """
    def decorator(func: Callable[["Config"], pd.DataFrame]):
        if name in DATA_LOADERS:
            raise ValueError(f"Data loader '{name}' is already registered")
        DATA_LOADERS[name] = func
        return func
    return decorator


def get_loader(name: str) -> Callable[["Config"], pd.DataFrame]:
    """Get a registered data loader by name.

    Args:
        name: Name of the loader to retrieve

    Returns:
        The loader function

    Raises:
        ValueError: If loader name is not registered
    """
    if name not in DATA_LOADERS:
        available = ", ".join(sorted(DATA_LOADERS.keys()))
        raise ValueError(
            f"Unknown data loader: '{name}'. "
            f"Available loaders: {available if available else '(none)'}"
        )
    return DATA_LOADERS[name]


def list_loaders() -> list[str]:
    """Get list of all registered loader names.

    Returns:
        Sorted list of loader names
    """
    return sorted(DATA_LOADERS.keys())


# Import all loader modules to trigger registration
# Add new loaders here
from . import catalan_gencat  # noqa: E402, F401
