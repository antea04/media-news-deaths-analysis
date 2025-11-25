"""Data loader registry for media deaths analysis.

This module provides a registry pattern for data loaders. Each loader is responsible
for fetching and formatting mortality data from a specific source.

Each loader must:
1. Be decorated with @register_loader("loader_name")
2. Return a pandas DataFrame with columns: ['code', 'cause', 'deaths', 'year']

Loaders are defined in the configs/data_loaders/ directory and automatically
registered when imported.
"""

from typing import Callable
import pandas as pd

# Registry of available data loaders
DATA_LOADERS: dict[str, Callable[[], pd.DataFrame]] = {}


def register_loader(name: str):
    """Decorator to register a data loader function.

    Args:
        name: Unique identifier for the loader (referenced in config files)

    Example:
        @register_loader("catalan_gencat")
        def load_data() -> pd.DataFrame:
            # Implementation here
            pass
    """
    def decorator(func: Callable[[], pd.DataFrame]):
        if name in DATA_LOADERS:
            raise ValueError(f"Data loader '{name}' is already registered")
        DATA_LOADERS[name] = func
        return func
    return decorator


def get_loader(name: str) -> Callable[[], pd.DataFrame]:
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


def discover_loaders():
    """Discover and import all data loaders from configs/data_loaders/.

    This function imports all loader modules to trigger their registration.
    Should be called once at startup.
    """
    try:
        # Import all loader modules to trigger registration
        from configs.data_loaders import catalan_gencat  # noqa: F401
        # Add new loaders here as they are created
    except ImportError as e:
        # If configs.data_loaders doesn't exist, that's okay
        # (allows library to work without loaders being present)
        pass
