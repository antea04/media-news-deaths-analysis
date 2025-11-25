#!/usr/bin/env python3
"""
Multi-country Media Deaths Analysis

This script runs the media deaths analysis for different countries using
country-specific configurations.

Usage:
    python run_analysis.py --country usa
    python run_analysis.py --country bra --restart
    python run_analysis.py --country usa --year 2022
"""

import argparse
import importlib
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import mediacloud.api
from dotenv import load_dotenv

import media_deaths_analysis
from media_deaths_analysis import (
    format_death_data,
    analyze_data,
    plot_media_deaths_matplotlib,
    query_results,
)
from query_generation import create_queries_by_cause
from load_who_data import load_who_deaths


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_API_SLEEP = 10  # seconds between API calls


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

def load_config(country_code):
    """Load configuration for specified country using ISO code."""
    country_code_lower = country_code.lower()

    # Try to import the config module dynamically
    try:
        config_module = importlib.import_module(f"configs.{country_code_lower}")
        return config_module.CONFIG
    except (ImportError, AttributeError) as e:
        # List available configs
        configs_dir = Path("configs")
        available_configs = sorted([
            f.stem for f in configs_dir.glob("*.py")
            if f.stem not in ("__init__", "TEMPLATE")
        ])

        raise ValueError(
            f"Config not found for country code '{country_code}'.\n"
            f"Available: {', '.join(available_configs)}\n"
            f"To add: create configs/{country_code_lower}.py with CONFIG dict"
        ) from e


def load_queries(config):
    """Load queries from configuration."""
    if "queries" not in config:
        raise ValueError(
            f"Config for {config['country_name']} is missing 'queries' section.\n"
            f"Please add query keywords to configs/{config['country_code'].lower()}.py"
        )
    return config["queries"]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_death_data(config, cache_file):
    """Load death data from cache or source."""
    if cache_file.exists():
        print(f"Loading cached death data from {cache_file}...")
        death_df = pd.read_csv(cache_file)
        print(f"Loaded {len(death_df)} causes from cache")
        return death_df

    print("Loading deaths data...")
    year = config["year"]

    if config["death_data_source"] == "cdc_wonder":
        leading_causes_df = pd.read_csv(
            config["death_data_urls"]["leading_causes"],
            sep="\t",
            storage_options={"User-Agent": "Mozilla/5.0"},
        )
        external_causes_df = pd.read_csv(
            config["death_data_urls"]["external_causes"],
            storage_options={"User-Agent": "Mozilla/5.0"},
        )

        # Set required global variables for legacy function
        media_deaths_analysis.YEAR = year
        media_deaths_analysis.TERRORISM_DEATHS_2023 = config["terrorism_deaths"]

        death_df = format_death_data(leading_causes_df, external_causes_df)

    elif config["death_data_source"] == "who":
        death_df = load_who_deaths(
            country_code=config["country_code"],
            year=year,
            causes_of_death=config["causes_of_death"],
            terrorism_deaths=config["terrorism_deaths"],
        )
    else:
        raise ValueError(f"Unknown death data source: {config['death_data_source']}")

    print(f"Loaded death data for {len(death_df)} causes")

    # Cache for future use
    death_df.to_csv(cache_file, index=False)
    print(f"Cached death data to {cache_file}")

    return death_df


def check_cache_completeness(cache_file, config):
    """Check if cached mentions data is complete."""
    if not cache_file.exists():
        return False

    cached_df = pd.read_csv(cache_file)
    expected_sources = {o["full_name"] for o in config["outlets"]} | {
        c["full_name"] for c in config["collections"]
    }
    cached_sources = set(cached_df["source"].unique())

    if expected_sources != cached_sources:
        missing = expected_sources - cached_sources
        print(f"📊 Cache incomplete: missing sources {missing}")
        return False

    # Check if each source has all causes
    expected_causes = set(config["causes_of_death"])
    for source in expected_sources:
        source_df = cached_df[cached_df["source"] == source]
        source_causes = set(source_df["cause"].unique())
        if expected_causes != source_causes:
            missing = expected_causes - source_causes
            print(f"📊 Cache incomplete for {source}: missing {missing}")
            return False

    return True


def query_media_cloud(config, cache_file):
    """Query Media Cloud API for media mentions."""
    # Load and validate queries
    raw_queries = load_queries(config)
    queries_to_use = {
        cause: raw_queries[cause]
        for cause in config["causes_of_death"]
        if cause in raw_queries
    }

    if len(queries_to_use) != len(config["causes_of_death"]):
        missing = set(config["causes_of_death"]) - set(queries_to_use.keys())
        raise ValueError(
            f"Missing queries for causes: {missing}\n"
            f"Please add them to the 'queries' section in configs/{config['country_code'].lower()}.py"
        )

    # Convert to query strings
    str_queries = create_queries_by_cause(queries_to_use)

    # Initialize API
    load_dotenv()
    api_token = os.getenv("MC_API_TOKEN")
    if not api_token:
        raise ValueError(
            "MC_API_TOKEN not set. Get API key from https://www.mediacloud.org/"
        )
    search_api = mediacloud.api.SearchApi(api_token)

    # Prepare sources
    all_sources = []
    for outlet in config["outlets"]:
        all_sources.append({
            "name": outlet["full_name"],
            "source_ids": [outlet["id"]],
            "collection_ids": None,
        })
    for collection in config["collections"]:
        all_sources.append({
            "name": collection["full_name"],
            "source_ids": [],
            "collection_ids": [collection["id"]],
        })

    # Load existing results for resume capability
    if cache_file.exists():
        print(f"📂 Loading existing cache from {cache_file}...")
        existing_df = pd.read_csv(cache_file)
        mentions_ls = existing_df.to_dict("records")
        completed_pairs = {(row["source"], row["cause"]) for row in mentions_ls}
        print(f"   Found {len(mentions_ls)} existing results, will skip these...")
        print()
    else:
        mentions_ls = []
        completed_pairs = set()

    # Query API
    total_sources = len(all_sources)
    total_causes = len(str_queries)
    api_sleep = config.get("api_sleep", DEFAULT_API_SLEEP)
    year = config["year"]
    start_time_overall = time.time()

    for source_idx, source in enumerate(all_sources, 1):
        print(f"\n[{source_idx}/{total_sources}] Querying {source['name']}...")

        for cause_idx, (cause_name, query) in enumerate(str_queries.items(), 1):
            # Skip if already cached
            if (source["name"], cause_name) in completed_pairs:
                print(f"  [{cause_idx}/{total_causes}] {cause_name}: ✓ already cached, skipping")
                continue

            time.sleep(api_sleep)
            start_time = time.time()

            try:
                cnt = query_results(
                    search_api,
                    query,
                    source["source_ids"],
                    year,
                    collection_ids=source["collection_ids"]
                )
            except Exception as e:
                print(f"  ⚠️  ERROR querying {cause_name} for {source['name']}: {e}")
                cnt = 0

            # Print progress
            elapsed = time.time() - start_time_overall
            minutes = int(elapsed / 60)
            seconds = int(elapsed % 60)
            warning = " ⚠️ " if cnt == 0 else ""
            print(
                f"  [{cause_idx}/{total_causes}] {cause_name}: {cnt} mentions{warning} "
                f"(retrieved in {time.time() - start_time:.1f}s, "
                f"total elapsed: {minutes:02d}:{seconds:02d})"
            )

            # Append result and save incrementally
            mentions_ls.append({
                "cause": cause_name,
                "mentions": cnt,
                "source": source["name"],
                "year": year,
            })
            pd.DataFrame(mentions_ls).to_csv(cache_file, index=False)

        print(f"  ✓ {source['name']} complete ({source_idx}/{total_sources} sources done)")

    # Final save
    mentions_df = pd.DataFrame(mentions_ls)
    mentions_df.to_csv(cache_file, index=False)
    print(f"\n✓ All queries complete! Final data saved to {cache_file}")
    return mentions_df


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_analysis(config, restart=False, run_single_queries=False,
                 use_saved_results=False, overwrite=True):
    """Run the full media deaths analysis for a given configuration."""

    country_code = config["country_code"].lower()
    language = config["language"]
    year = config["year"]

    print("=" * 80)
    print("MEDIA DEATHS ANALYSIS")
    print("=" * 80)
    print(f"Country: {config['country_name']} ({country_code.upper()})")
    print(f"Language: {language}")
    print(f"Year: {year}")
    print(f"Outlets: {[o['full_name'] for o in config['outlets']]}")
    print(f"Restart from scratch: {restart}")
    print(f"Use saved results: {use_saved_results}")
    print("=" * 80)
    print()

    # Setup paths
    output_dir = Path(f"./data/{country_code}")
    output_dir.mkdir(parents=True, exist_ok=True)

    death_data_file = output_dir / f"death_data_{year}.csv"
    mentions_file = output_dir / f"media_deaths_mentions_{language}.csv"
    results_file = output_dir / f"media_deaths_results_{language}.csv"
    plot_file = output_dir / "media_deaths_by_source.png"

    # Handle restart flag - clear existing cache
    if restart and mentions_file.exists():
        print(f"🔄 Restart flag set - removing existing cache at {mentions_file}")
        mentions_file.unlink()
        print()

    # Load or compute results
    if use_saved_results:
        print(f"Loading saved results from {results_file}...")
        if not results_file.exists():
            raise FileNotFoundError(
                f"Saved results not found at {results_file}. Run without --use-saved-results first."
            )
        media_deaths_df = pd.read_csv(results_file)
    else:
        # Load death data
        death_df = load_death_data(config, death_data_file)
        print()

        # Load or query media mentions
        cache_is_complete = check_cache_completeness(mentions_file, config)

        if cache_is_complete:
            print(f"✓ Using complete cached data from {mentions_file}")
            mentions_df = pd.read_csv(mentions_file)
        else:
            if mentions_file.exists():
                print("📂 Resuming from incomplete cache...")
            else:
                print("🔄 Starting fresh query run...")
            print("Querying Media Cloud API...")
            print("This may take ~30 minutes due to API rate limits...")
            print()
            mentions_df = query_media_cloud(config, mentions_file)

        print()

        # Analyze data
        print("Analyzing data...")

        # Set required global variables for legacy function
        media_deaths_analysis.CAUSES_OF_DEATH = config["causes_of_death"]
        media_deaths_analysis.OUTLETS = config["outlets"]
        media_deaths_analysis.COLLECTIONS = config["collections"]

        media_deaths_df = analyze_data(mentions_df, death_df, run_single_queries)

        # Save results
        if overwrite:
            media_deaths_df.to_csv(results_file, index=False)
            print(f"Saved analysis results to {results_file}")
        print()

    # Display summary
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    summary_cols = ["cause", "deaths"]
    for outlet in config["outlets"]:
        summary_cols.append(f"{outlet['short_name']}_mentions")
    for collection in config["collections"]:
        summary_cols.append(f"{collection['short_name']}_mentions")

    print(media_deaths_df[summary_cols].to_string(index=False))
    print()

    # Create visualizations
    print("=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    print()
    print("Media mentions by source...")

    plot_cols = ["deaths_share"]
    plot_labels = ["Deaths"]

    for outlet in config["outlets"]:
        plot_cols.append(f"{outlet['short_name']}_share")
        plot_labels.append(outlet['short_name'].upper())

    for collection in config["collections"]:
        plot_cols.append(f"{collection['short_name']}_share")
        plot_labels.append(collection['full_name'])

    # Set required global variables for legacy function
    media_deaths_analysis.YEAR = year
    media_deaths_analysis.CAUSES_OF_DEATH = config["causes_of_death"]
    media_deaths_analysis.FIXED_COLOURS = config["colors"]

    plot_media_deaths_matplotlib(
        media_deaths_df,
        columns=plot_cols,
        bar_labels=plot_labels,
        absolute=False,
        title=f"Media mentions of causes of death in {year} - {config['country_name']}",
        save_path=plot_file,
        fixed_colors=config["colors"],
    )

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}/")
    print()
    print("Press Enter to exit...")
    input()


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Media Deaths Analysis - Multi-country support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py --country usa
  python run_analysis.py --country bra --restart
  python run_analysis.py --country usa --use-saved-results
  python run_analysis.py --country usa --year 2022

Adding new countries:
  1. Create configs/[country_code].py with configuration (use ISO 3-letter code)
  2. Add translated query keywords to the config
  3. Run: python run_analysis.py --country [country_code]
        """
    )

    parser.add_argument("--country", required=True, help="Country code (usa, bra, etc.)")
    parser.add_argument("--year", type=int, help="Year to analyze (overrides config default)")
    parser.add_argument("--restart", action="store_true",
                        help="Start from scratch (clear cached mentions data)")
    parser.add_argument("--run-single-queries", action="store_true",
                        help="Run single-keyword queries")
    parser.add_argument("--use-saved-results", action="store_true",
                        help="Use saved final results (skip all computation)")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Don't overwrite existing result files")

    args = parser.parse_args()

    try:
        config = load_config(args.country)
        if args.year:
            config["year"] = args.year

        run_analysis(
            config=config,
            restart=args.restart,
            run_single_queries=args.run_single_queries,
            use_saved_results=args.use_saved_results,
            overwrite=not args.no_overwrite,
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
