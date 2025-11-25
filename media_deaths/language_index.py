"""Language-region index resolution for media deaths analysis.

This module handles mapping language-region codes (e.g., 'ca_ES', 'en_US')
to configuration files by scanning the configs directory.
"""

from pathlib import Path
import yaml


def scan_config_directory(configs_dir: Path = None) -> dict:
    """Scan configs directory and build index from config files.

    Args:
        configs_dir: Path to configs directory. Defaults to configs/

    Returns:
        Dictionary mapping language codes to config info

    Raises:
        FileNotFoundError: If configs directory doesn't exist
    """
    if configs_dir is None:
        # Default to configs/ relative to this file's parent
        configs_dir = Path(__file__).parent.parent / "configs"

    if not configs_dir.exists():
        raise FileNotFoundError(f"Configs directory not found: {configs_dir}")

    index = {}

    # Scan all .yml files in configs directory
    for config_file in configs_dir.glob("*.yml"):
        # The language-region code is the filename without extension
        code = config_file.stem

        # Try to load metadata from the config file
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            metadata = config.get("metadata", {})
            index[code] = {
                "name": metadata.get("name", code),
                "description": metadata.get("description", ""),
                "config_path": config_file,
            }
        except Exception as e:
            # If we can't load the file, skip it
            print(f"Warning: Could not load config {config_file}: {e}")
            continue

    return index


def resolve_language_code(code: str, configs_dir: Path = None) -> Path:
    """Resolve a language-region code to its configuration file path.

    Args:
        code: Language-region code (e.g., 'ca_ES', 'en_US')
        configs_dir: Optional path to configs directory

    Returns:
        Absolute path to configuration file

    Raises:
        ValueError: If language-region code is not found
        FileNotFoundError: If configs directory doesn't exist
    """
    index = scan_config_directory(configs_dir)

    if code not in index:
        available = list_available_languages(configs_dir)
        raise ValueError(
            f"Unknown language-region code: '{code}'\n"
            f"Available codes: {', '.join(available)}"
        )

    config_path = index[code]["config_path"]

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found for language-region '{code}': {config_path}"
        )

    return config_path.resolve()


def list_available_languages(configs_dir: Path = None) -> list[str]:
    """List all available language-region codes.

    Args:
        configs_dir: Optional path to configs directory

    Returns:
        Sorted list of language-region codes
    """
    try:
        index = scan_config_directory(configs_dir)
        return sorted(index.keys())
    except FileNotFoundError:
        return []


def format_available_languages(configs_dir: Path = None) -> str:
    """Format available language-region codes for display.

    Args:
        configs_dir: Optional path to configs directory

    Returns:
        Formatted string listing language-region codes with descriptions
    """
    try:
        index = scan_config_directory(configs_dir)

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
        return "Configs directory not found."
