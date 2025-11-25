#!/usr/bin/env python3
"""CLI entry point for media-deaths analysis."""

import argparse
from pathlib import Path

from media_deaths.config import Config
from media_deaths.main import main


def cli():
    """Command-line interface for media deaths analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze media mentions of causes of death compared to actual mortality data"
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to configuration YAML file (e.g., config.yml)",
    )
    parser.add_argument(
        "--causes",
        type=str,
        nargs="+",
        help="Specific causes of death to analyze (space-separated). If not provided, analyzes all causes from config.",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {args.config_file}")
        return 1

    config = Config(config_path)

    # Run analysis
    main(config, causes_of_death=args.causes)

    return 0


if __name__ == "__main__":
    exit(cli())
