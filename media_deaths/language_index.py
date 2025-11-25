"""Language-region index resolution for media deaths analysis.

This module handles mapping language-region codes (e.g., 'ca_ES', 'en_US')
to configuration files.
"""

from pathlib import Path
import yaml


def load_language_index(index_path: Path = None) -> dict:
    """Load the language index YAML file.

    Args:
        index_path: Path to index.yml file. Defaults to configs/index.yml

    Returns:
        Dictionary mapping language codes to config info

    Raises:
        FileNotFoundError: If index file doesn't exist
    """
    if index_path is None:
        # Default to configs/index.yml relative to this file's parent
        index_path = Path(__file__).parent.parent / "configs" / "index.yml"

    if not index_path.exists():
        raise FileNotFoundError(f"Language index not found: {index_path}")

    with open(index_path, "r") as f:
        index = yaml.safe_load(f)

    return index or {}


def resolve_language_code(code: str, index_path: Path = None) -> Path:
    """Resolve a language-region code to its configuration file path.

    Args:
        code: Language-region code (e.g., 'ca_ES', 'en_US')
        index_path: Optional path to index.yml

    Returns:
        Absolute path to configuration file

    Raises:
        ValueError: If language-region code is not found in index
        FileNotFoundError: If index file doesn't exist
    """
    index = load_language_index(index_path)

    if code not in index:
        available = list_available_languages(index_path)
        raise ValueError(
            f"Unknown language-region code: '{code}'\n"
            f"Available codes: {', '.join(available)}"
        )

    # Get config file path relative to index file
    if index_path is None:
        index_path = Path(__file__).parent.parent / "configs" / "index.yml"

    config_file = index[code].get("config")
    if not config_file:
        raise ValueError(f"Language-region code '{code}' has no config file specified")

    # Resolve relative to configs directory
    config_path = index_path.parent / config_file

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found for language-region '{code}': {config_path}"
        )

    return config_path.resolve()


def list_available_languages(index_path: Path = None) -> list[str]:
    """List all available language-region codes.

    Args:
        index_path: Optional path to index.yml

    Returns:
        Sorted list of language-region codes
    """
    try:
        index = load_language_index(index_path)
        return sorted(index.keys())
    except FileNotFoundError:
        return []


def format_available_languages(index_path: Path = None) -> str:
    """Format available language-region codes for display.

    Args:
        index_path: Optional path to index.yml

    Returns:
        Formatted string listing language-region codes with descriptions
    """
    try:
        index = load_language_index(index_path)

        if not index:
            return "No language-region codes configured."

        lines = ["Available language-region codes:"]
        for code in sorted(index.keys()):
            lang_info = index[code]
            name = lang_info.get("name", code)
            desc = lang_info.get("description", "")
            if desc:
                lines.append(f"  {code:10} - {name} ({desc})")
            else:
                lines.append(f"  {code:10} - {name}")

        return "\n".join(lines)
    except FileNotFoundError:
        return "Language-region index file not found."
