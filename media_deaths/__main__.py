#!/usr/bin/env python3
"""CLI entry point for media-deaths analysis."""

import argparse
from pathlib import Path

from media_deaths.config import Config
from media_deaths.main import main
from media_deaths.language_index import (
    resolve_language_code,
    format_available_languages,
)


def cli():
    """Command-line interface for media deaths analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze media mentions of causes of death compared to actual mortality data",
        epilog="Examples:\n"
        "  media-deaths ca_ES                   # Use Catalan (Spain) configuration\n"
        "  media-deaths --custom-config my.yml  # Use custom config file\n"
        "  media-deaths --list-languages        # Show available language-region codes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "language_code",
        type=str,
        nargs="?",
        help="Language-region code (e.g., 'ca_ES' for Catalan in Spain)",
    )
    parser.add_argument(
        "--custom-config",
        type=str,
        help="Path to custom configuration YAML file",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List available language-region codes and exit",
    )
    parser.add_argument(
        "--causes",
        type=str,
        nargs="+",
        help="Specific causes of death to analyze (space-separated). If not provided, analyzes all causes from config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory for output files (default: ./data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running queries",
    )

    args = parser.parse_args()

    # Handle --list-languages flag
    if args.list_languages:
        print(format_available_languages())
        return 0

    # Determine config file path
    config_path = None

    if args.custom_config:
        # Use custom config file
        config_path = Path(args.custom_config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {args.custom_config}")
            return 1
    elif args.language_code:
        # Resolve language code to config file
        try:
            config_path = resolve_language_code(args.language_code)
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}")
            print()
            print(format_available_languages())
            return 1
    else:
        # Neither provided - show error and available languages
        print("Error: Please provide either a language-region code or --custom-config")
        print()
        print(format_available_languages())
        print()
        print("Usage:")
        print("  media-deaths <language_region_code>")
        print("  media-deaths --custom-config <config_file>")
        print("  media-deaths --list-languages")
        return 1

    # Load configuration
    config = Config(config_path)

    # Set output directory
    output_dir = Path(args.output_dir)
    config.OUTPUT_DIR = output_dir

    # Run analysis (or dry run)
    main(config, causes_of_death=args.causes, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    exit(cli())
