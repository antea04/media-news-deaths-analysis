#!/usr/bin/env python3
"""
Multi-country Media Deaths Analysis

This script runs the media deaths analysis for different countries using
country-specific configurations.

Usage:
    python run_analysis.py --country usa
    python run_analysis.py --country usa --rerun-queries
    python run_analysis.py --country brazil --year 2020

The script imports all the core functions from media_deaths_analysis.py
and uses country configs from the configs/ folder.
"""

import argparse
import os
import sys
from pathlib import Path

# Import all the functions we need from the original script
from media_deaths_analysis import (
    format_death_data,
    get_media_mentions,
    analyze_data,
    plot_media_deaths_matplotlib,
)

# Import query building utilities
from query_generation import create_queries_by_cause

# Import pandas for data manipulation
import pandas as pd


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

def load_config(country_code):
    """
    Load configuration for specified country.

    Args:
        country_code: Country code (usa, brazil, spain, etc.)

    Returns:
        dict: Configuration dictionary
    """
    # Import available configs
    from configs import usa, brazil

    configs = {
        "usa": usa.CONFIG,
        "brazil": brazil.CONFIG,
    }

    if country_code.lower() not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(
            f"Unknown country '{country_code}'. Available: {available}\n"
            f"To add a new country, create configs/{country_code}.py"
        )

    return configs[country_code.lower()]


def load_queries(config):
    """
    Load queries from configuration.

    Args:
        config: Configuration dictionary containing "queries" key

    Returns:
        dict: Raw query dictionaries for each cause
    """
    if "queries" not in config:
        raise ValueError(
            f"Config for {config['country_name']} is missing 'queries' section.\n"
            f"Please add query keywords to configs/{config['country_code'].lower()}.py"
        )

    return config["queries"]


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_analysis(
    config,
    rerun_queries=False,
    run_single_queries=False,
    use_saved_results=False,
    overwrite=True,
    verbose=True,
):
    """
    Run the full media deaths analysis for a given configuration.

    Args:
        config: Configuration dictionary
        rerun_queries: Whether to re-query Media Cloud API
        run_single_queries: Whether to run single-keyword queries
        use_saved_results: Whether to use cached final results
        overwrite: Whether to overwrite existing files
        verbose: Whether to print progress messages
    """

    # Extract config values
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
    print(f"Rerun queries: {rerun_queries}")
    print(f"Use saved results: {use_saved_results}")
    print("=" * 80)
    print()

    # Create output directory for this country
    output_dir = Path(f"./data/{country_code}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # File paths for caching
    death_data_file = output_dir / f"death_data_{year}.csv"
    mentions_file = output_dir / f"media_deaths_mentions_{language}.csv"
    results_file = output_dir / f"media_deaths_results_{language}.csv"
    plot_file = output_dir / "media_deaths_by_source.png"

    # Load or use saved results
    if use_saved_results:
        print(f"Loading saved results from {results_file}...")
        if not results_file.exists():
            raise FileNotFoundError(
                f"Saved results not found at {results_file}. "
                "Run with --rerun-queries first."
            )
        media_deaths_df = pd.read_csv(results_file)
    else:
        # Step 1: Load death data (with caching)
        if death_data_file.exists():
            print(f"Loading cached death data from {death_data_file}...")
            death_df = pd.read_csv(death_data_file)
            print(f"Loaded {len(death_df)} causes from cache")
        else:
            print("Loading deaths data...")
            if config["death_data_source"] == "cdc_wonder":
                # Load CDC data
                leading_causes_df = pd.read_csv(
                    config["death_data_urls"]["leading_causes"],
                    sep="\t",
                    storage_options={"User-Agent": "Mozilla/5.0"},
                )
                external_causes_df = pd.read_csv(
                    config["death_data_urls"]["external_causes"],
                    storage_options={"User-Agent": "Mozilla/5.0"},
                )

                # Override global variables needed by format_death_data
                import media_deaths_analysis as mda
                mda.YEAR = year
                mda.TERRORISM_DEATHS_2023 = config["terrorism_deaths"]

                death_df = format_death_data(leading_causes_df, external_causes_df)
            elif config["death_data_source"] == "who":
                # Load WHO mortality data from OWID catalog
                from load_who_data import load_who_deaths

                death_df = load_who_deaths(
                    country_code=config["country_code"],
                    year=year,
                    causes_of_death=config["causes_of_death"],
                    terrorism_deaths=config["terrorism_deaths"],
                )
            else:
                raise ValueError(f"Unknown death data source: {config['death_data_source']}")

            print(f"Loaded death data for {len(death_df)} causes")

            # Cache the death data immediately
            death_df.to_csv(death_data_file, index=False)
            print(f"Cached death data to {death_data_file}")

        print()

        # Step 2: Get media mentions (query Media Cloud or load cached)
        # Smart detection: check if cache is complete
        cache_is_complete = False
        if mentions_file.exists() and not rerun_queries:
            cached_df = pd.read_csv(mentions_file)
            expected_sources = {o["full_name"] for o in config["outlets"]} | {c["full_name"] for c in config["collections"]}
            cached_sources = set(cached_df["source"].unique())
            missing_sources = expected_sources - cached_sources

            if not missing_sources:
                # Cache has all sources - check if each source has all causes
                expected_causes = set(config["causes_of_death"])
                cache_is_complete = True
                for source in expected_sources:
                    source_df = cached_df[cached_df["source"] == source]
                    source_causes = set(source_df["cause"].unique())
                    missing_causes = expected_causes - source_causes
                    if missing_causes:
                        print(f"📊 Cache incomplete for {source}: missing {missing_causes}")
                        cache_is_complete = False
                        break
            else:
                print(f"📊 Cache incomplete: missing sources {missing_sources}")

        # Decide whether to query
        if cache_is_complete:
            print(f"✓ Using complete cached data from {mentions_file}")
            mentions_df = pd.read_csv(mentions_file)
        else:
            # Need to query (either no cache, incomplete cache, or --rerun-queries flag)
            if mentions_file.exists() and not rerun_queries:
                print(f"📂 Resuming from incomplete cache...")
            elif rerun_queries:
                print(f"🔄 Re-running queries (--rerun-queries flag set)...")
            print("Querying Media Cloud API...")
            print("This may take ~30 minutes due to API rate limits...")
            print()

            # Load queries from config
            raw_queries = load_queries(config)

            # Filter to only the causes we're analyzing
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

            # Query Media Cloud with incremental saving
            import mediacloud.api
            from dotenv import load_dotenv
            load_dotenv()
            MC_API_TOKEN = os.getenv("MC_API_TOKEN")

            if not MC_API_TOKEN:
                raise ValueError(
                    "MC_API_TOKEN not set. Get API key from https://www.mediacloud.org/ "
                    "or set RERUN_QUERIES=False to use cached data"
                )

            # Initialize search API
            search_api = mediacloud.api.SearchApi(MC_API_TOKEN)

            # Import helper functions
            from media_deaths_analysis import query_results
            import time

            # Prepare all sources (outlets + collections)
            all_sources = []
            for outlet in config["outlets"]:
                all_sources.append({
                    "type": "outlet",
                    "name": outlet["full_name"],
                    "source_ids": [outlet["id"]],
                    "collection_ids": None,
                })
            for collection in config["collections"]:
                all_sources.append({
                    "type": "collection",
                    "name": collection["full_name"],
                    "source_ids": [],
                    "collection_ids": [collection["id"]],
                })

            total_sources = len(all_sources)
            total_causes = len(str_queries)
            api_sleep = config.get("api_sleep", 10)

            # Load existing cache if resuming (whether or not --rerun-queries is set)
            if mentions_file.exists():
                print(f"📂 Loading existing cache from {mentions_file}...")
                existing_df = pd.read_csv(mentions_file)
                mentions_ls = existing_df.to_dict("records")

                # Track which (source, cause) pairs we already have
                completed_pairs = {(row["source"], row["cause"]) for row in mentions_ls}
                print(f"   Found {len(mentions_ls)} existing results, will skip these...")
                print()
            else:
                mentions_ls = []
                completed_pairs = set()

            # Query each source and cause with incremental saving
            start_time_overall = time.time()

            for source_idx, source in enumerate(all_sources, 1):
                print(f"\n[{source_idx}/{total_sources}] Querying {source['name']}...")

                for cause_idx, (cause_name, query) in enumerate(str_queries.items(), 1):
                    # Skip if we already have this (source, cause) pair
                    if (source["name"], cause_name) in completed_pairs:
                        print(f"  [{cause_idx}/{total_causes}] {cause_name}: ✓ already cached, skipping")
                        continue

                    time.sleep(api_sleep)  # Rate limiting
                    start_time = time.time()

                    # Query this specific cause
                    try:
                        # Debug: print query details for Folha's first query
                        if "Folha" in source["name"] and cause_idx == 1:
                            print(f"      DEBUG - Query: {query[:150]}...")
                            print(f"      DEBUG - Source IDs: {source['source_ids']}")
                            print(f"      DEBUG - Year: {year}")
                            print(f"      DEBUG - Collection IDs: {source['collection_ids']}")

                        cnt = query_results(
                            search_api,
                            query,
                            source["source_ids"],
                            year,
                            collection_ids=source["collection_ids"]
                        )
                    except Exception as e:
                        print(f"  ⚠️  ERROR querying {cause_name} for {source['name']}: {e}")
                        print(f"      Query: {query}")
                        print(f"      Source IDs: {source['source_ids']}")
                        cnt = 0  # Record as 0 but continue

                    # Print progress
                    elapsed = time.time() - start_time_overall
                    minutes = int(elapsed / 60)
                    seconds = int(elapsed % 60)

                    # Warn if count is 0 (might indicate a problem)
                    warning = " ⚠️ " if cnt == 0 else ""
                    print(f"  [{cause_idx}/{total_causes}] {cause_name}: {cnt} mentions{warning} "
                          f"(retrieved in {time.time() - start_time:.1f}s, "
                          f"total elapsed: {minutes:02d}:{seconds:02d})")

                    # Append result
                    mentions_ls.append({
                        "cause": cause_name,
                        "mentions": cnt,
                        "source": source["name"],
                        "year": year,
                    })

                    # Save progress after EACH cause (every ~10 seconds)
                    temp_df = pd.DataFrame(mentions_ls)
                    temp_df.to_csv(mentions_file, index=False)

                print(f"  ✓ {source['name']} complete ({source_idx}/{total_sources} sources done)")

            # Final save
            mentions_df = pd.DataFrame(mentions_ls)
            mentions_df.to_csv(mentions_file, index=False)
            print(f"\n✓ All queries complete! Final data saved to {mentions_file}")

        print()

        # Step 3: Analyze data
        print("Analyzing data...")

        # Override global variables for analyze_data
        import media_deaths_analysis as mda
        mda.CAUSES_OF_DEATH = config["causes_of_death"]
        mda.OUTLETS = config["outlets"]
        mda.COLLECTIONS = config["collections"]

        media_deaths_df = analyze_data(
            mentions_df,
            death_df,
            run_single_queries=run_single_queries
        )

        # Save results
        if overwrite:
            media_deaths_df.to_csv(results_file, index=False)
            print(f"Saved analysis results to {results_file}")
        print()

    # Step 4: Display summary
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Build columns list dynamically based on available outlets
    summary_cols = ["cause", "deaths"]
    for outlet in config["outlets"]:
        summary_cols.append(f"{outlet['short_name']}_mentions")
    for collection in config["collections"]:
        summary_cols.append(f"{collection['short_name']}_mentions")

    print(media_deaths_df[summary_cols].to_string(index=False))
    print()

    # Step 5: Create visualizations
    print("=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    print()

    print("Media mentions by source...")

    # Build columns and labels for plotting
    plot_cols = ["deaths_share"]
    plot_labels = ["Deaths"]

    for outlet in config["outlets"]:
        plot_cols.append(f"{outlet['short_name']}_share")
        plot_labels.append(outlet['short_name'].upper())

    for collection in config["collections"]:
        plot_cols.append(f"{collection['short_name']}_share")
        plot_labels.append(collection['full_name'])

    # Override global variables for plotting
    import media_deaths_analysis as mda
    mda.YEAR = year
    mda.CAUSES_OF_DEATH = config["causes_of_death"]
    mda.FIXED_COLOURS = config["colors"]

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
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for command-line usage."""

    parser = argparse.ArgumentParser(
        description="Media Deaths Analysis - Multi-country support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run USA analysis (uses cached data if available)
  python run_analysis.py --country usa

  # Re-query Media Cloud API for USA
  python run_analysis.py --country usa --rerun-queries

  # Use saved final results (skip all computation)
  python run_analysis.py --country usa --use-saved-results

  # Run for a different year (if config supports it)
  python run_analysis.py --country usa --year 2022

Adding new countries:
  1. Create configs/[country].py with configuration
  2. Create queries/[language].py with translated keywords
  3. Run: python run_analysis.py --country [country]
        """
    )

    parser.add_argument(
        "--country",
        required=True,
        help="Country code (usa, brazil, etc.)",
    )

    parser.add_argument(
        "--year",
        type=int,
        help="Year to analyze (overrides config default)",
    )

    parser.add_argument(
        "--rerun-queries",
        action="store_true",
        help="Re-query Media Cloud API (takes ~30 min, otherwise uses cached data)",
    )

    parser.add_argument(
        "--run-single-queries",
        action="store_true",
        help="Also run single-keyword queries (in addition to multiple mentions)",
    )

    parser.add_argument(
        "--use-saved-results",
        action="store_true",
        help="Use saved final results (skip all computation)",
    )

    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Don't overwrite existing result files",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.country)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Override year if specified
    if args.year:
        config["year"] = args.year

    # Run analysis
    try:
        run_analysis(
            config=config,
            rerun_queries=args.rerun_queries,
            run_single_queries=args.run_single_queries,
            use_saved_results=args.use_saved_results,
            overwrite=not args.no_overwrite,
            verbose=True,
        )
    except Exception as e:
        print(f"\nError during analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
