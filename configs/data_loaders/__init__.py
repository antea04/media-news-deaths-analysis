"""Data loaders for media deaths analysis.

This module contains specific data loader implementations.
Each loader registers itself with the central registry in media_deaths.data_loaders.
"""

# Import all loader modules to trigger registration
from . import catalan_gencat  # noqa: F401
