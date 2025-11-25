#!/usr/bin/env python3
"""
Media Deaths Analysis Script

This script contains all the code used to create, analyze, and visualize the data
for the article on mentions for causes of death in the media.

For more details, see the methodology document at:
https://docs.owid.io/projects/etl/analyses/media_deaths/methodology/
"""

import datetime as dt
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import mediacloud.api

from media_deaths.log import Log
from media_deaths.config import Config
from media_deaths.query_generation import (
    create_full_queries,
    create_single_keyword_queries,
)
from media_deaths.data_loaders import get_loader, discover_loaders

# Discover and register all available data loaders
discover_loaders()

# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main(
    config: Config, causes_of_death: list[str] | None = None, dry_run: bool = False
):
    """Main execution function.

    Args:
        config: Configuration object loaded from YAML file
        causes_of_death: Optional list of specific causes to analyze. If None, uses all from config.
        dry_run: If True, show what would be executed without running queries
    """
    # Validate causes of death
    if causes_of_death is None:
        causes_of_death = config.CAUSES_OF_DEATH_ALL
    else:
        config.check_valid_causes(causes_of_death)

    CAUSES_OF_DEATH = causes_of_death
    outlets = config.OUTLETS

    # Set up output directory structure: ./data/{code}/
    output_dir = config.OUTPUT_DIR / config.CODE
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate estimated time if running queries
    if config.RERUN_QUERIES and not config.USE_SAVED_RESULTS:
        num_causes = len(CAUSES_OF_DEATH)
        num_sources = len(outlets) + len(config.COLLECTIONS)
        num_queries = num_causes * num_sources
        # Each query takes ~0.5 minutes (30 seconds)
        estimated_minutes = int(num_queries * 0.5)
        time_estimate = f"~{estimated_minutes} min ({num_queries} queries)"
    else:
        time_estimate = "<1 min (cached)"

    # Prepare summary info
    summary_info = {
        "year": config.YEAR,
        "code": config.CODE,
        "outlets": ", ".join([outlet["full_name"] for outlet in outlets]),
        "rerun_queries": config.RERUN_QUERIES,
        "run_single_keyword_queries": config.RUN_SINGLE_QUERIES,
        "use_saved_results": config.USE_SAVED_RESULTS,
        "estimated_time": time_estimate,
    }

    if dry_run:
        summary_info["mode"] = "DRY RUN"

    Log.summary("MEDIA DEATHS ANALYSIS", **summary_info)

    # Handle dry run mode
    if dry_run:
        Log.info("Dry run mode - showing execution plan without running queries")
        print()

        # Show causes to analyze
        Log.section("Causes of Death")
        for i, cause in enumerate(CAUSES_OF_DEATH, 1):
            print(f"  {i}. {cause}")
        print()

        # Show sources to query
        Log.section("Media Sources")
        print(f"  Outlets ({len(outlets)}):")
        for outlet in outlets:
            print(f"    - {outlet['full_name']} (id: {outlet['id']})")
        print()
        print(f"  Collections ({len(config.COLLECTIONS)}):")
        for collection in config.COLLECTIONS:
            print(f"    - {collection['full_name']} (id: {collection['id']})")
        print()

        # Show query plan
        Log.section("Query Execution Plan")
        num_causes = len(CAUSES_OF_DEATH)
        num_sources = len(outlets) + len(config.COLLECTIONS)
        num_queries = num_causes * num_sources
        print(f"  Total queries: {num_queries}")
        print(f"  Formula: {num_causes} causes × {num_sources} sources")
        print(f"  Estimated time: ~{int(num_queries * 0.5)} minutes")
        print()

        # Show output files
        Log.section("Output Files")
        print(f"  Directory: {output_dir}")
        print(f"  - media_deaths_mentions.csv")
        print(f"  - media_deaths_results.csv")
        print(f"  - media_deaths_by_source.png")
        print()

        Log.success("Dry run completed. Run without --dry-run to execute analysis.")
        return

    # Load or use saved results
    if config.USE_SAVED_RESULTS:
        results_file = output_dir / "media_deaths_results.csv"
        Log.info(f"Loading saved results from {results_file}")
        media_deaths_df = pd.read_csv(results_file)
    else:
        # Load and format death data using configured loader
        data_loader = get_loader(config.DATA_LOADER)
        death_df = data_loader()

        # Validate data loader output
        _validate_death_data(death_df, config)
        print()

        # Get media mentions
        mentions_df = get_media_mentions(
            outlets=outlets,
            causes_of_death=CAUSES_OF_DEATH,
            api_token=config.MC_API_TOKEN,
            rerun_queries=config.RERUN_QUERIES,
            year=config.YEAR,
            api_sleep=config.API_SLEEP,
            verbose=config.VERBOSE,
            queries=config.QUERIES,
            collections=config.COLLECTIONS,
            overwrite=config.OVERWRITE,
            run_single_queries=config.RUN_SINGLE_QUERIES,
            output_dir=output_dir,
        )
        print()

        # Analyze data
        media_deaths_df = analyze_data(
            mentions_df=mentions_df,
            death_df=death_df,
            causes_of_death=CAUSES_OF_DEATH,
            outlets=config.OUTLETS,
            collections=config.COLLECTIONS,
            run_single_queries=config.RUN_SINGLE_QUERIES,
        )

        # Save results
        if config.OVERWRITE:
            results_file = output_dir / "media_deaths_results.csv"
            media_deaths_df.to_csv(results_file, index=False)
            Log.success(f"Saved analysis results to {results_file}")
        print()

    # Display summary statistics
    Log.section("SUMMARY STATISTICS")
    print(
        media_deaths_df[
            [
                "cause",
                "deaths",
            ]
            + [f"{o['short_name']}_mentions" for o in config.OUTLETS]
            + [f"{o['short_name']}_mentions" for o in config.COLLECTIONS]
        ].to_string(index=False)
    )
    print()

    # Create visualizations
    Log.section("CREATING VISUALIZATIONS")

    # 1. Media mentions by source
    Log.info("Generating media mentions by source plot...")
    plot_save_path = output_dir / "media_deaths_by_source.png"
    plot_media_deaths_matplotlib(
        media_deaths_df=media_deaths_df,
        causes_of_death=CAUSES_OF_DEATH,
        columns=[
            "deaths_share",
        ]
        + [f"{o['short_name']}_share" for o in config.OUTLETS]
        + [f"{o['short_name']}_share" for o in config.COLLECTIONS],
        bar_labels=[
            "Deaths",
        ]
        + [o["full_name"] for o in config.OUTLETS]
        + [o["full_name"] for o in config.COLLECTIONS],
        fixed_colors=config.FIXED_COLORS,
        year=config.YEAR,
        absolute=False,
        title=f"Media mentions of causes of death in {config.YEAR}",
        save_path=str(plot_save_path),
    )
    print()

    Log.section("ANALYSIS COMPLETE")
    Log.success("All tasks completed successfully")


# ===============================================================
# DATA VALIDATION FUNCTIONS
# ==============================================================


def _validate_death_data(death_df: pd.DataFrame, config: Config) -> None:
    """Validate that data loader output meets required specifications.

    Checks:
    1. DataFrame has required columns: 'cause', 'deaths', 'year'
    2. Year column contains only a single value matching config.YEAR
    3. Cause values are all present in config.CAUSES_OF_DEATH_ALL
    4. No duplicate (year, cause) entries exist

    Args:
        death_df: DataFrame returned by data loader
        config: Configuration object

    Raises:
        ValueError: If any validation check fails
    """
    # Check 1: Required columns exist
    required_columns = {"cause", "deaths", "year"}
    actual_columns = set(death_df.columns)
    missing_columns = required_columns - actual_columns

    if missing_columns:
        raise ValueError(
            f"Data loader output missing required columns: {missing_columns}. "
            f"Required columns: {required_columns}. "
            f"Actual columns: {actual_columns}"
        )

    # Check 2: Year column has single value matching config.YEAR
    unique_years = death_df["year"].unique()
    if len(unique_years) == 0:
        raise ValueError("Data loader returned empty year column")
    if len(unique_years) > 1:
        raise ValueError(
            f"Data loader returned multiple years: {sorted(unique_years)}. "
            f"Expected only: {config.YEAR}"
        )
    if unique_years[0] != config.YEAR:
        raise ValueError(
            f"Data loader year ({unique_years[0]}) does not match config.YEAR ({config.YEAR})"
        )

    # Check 3: All causes are valid (present in config)
    loader_causes = set(death_df["cause"].unique())
    valid_causes = set(config.CAUSES_OF_DEATH_ALL)
    invalid_causes = loader_causes - valid_causes

    if invalid_causes:
        raise ValueError(
            f"Data loader returned invalid causes: {invalid_causes}. "
            f"Valid causes from config: {valid_causes}"
        )

    # Check 4: No duplicate (year, cause) entries
    duplicate_check = death_df.groupby(["year", "cause"]).size()
    duplicates = duplicate_check[duplicate_check > 1]

    if len(duplicates) > 0:
        duplicate_entries = duplicates.to_dict()
        raise ValueError(
            f"Data loader returned duplicate (year, cause) entries: {duplicate_entries}"
        )

    Log.success(
        f"Data validation passed: {len(death_df)} rows, "
        f"{len(loader_causes)} causes for year {config.YEAR}"
    )


# ===============================================================
# DATA PROCESSING FUNCTIONS
# ==============================================================


def get_start_end(year):
    """Get start and end dates for a given year."""
    return (dt.date(year, 1, 1), dt.date(year, 12, 31))


def query_results(search_api, query, source_ids, year, collection_ids=None):
    """
    Helper function to use Media Cloud API.

    Args:
        search_api: Media Cloud API instance
        query: Search query string
        source_ids: List of source IDs
        year: Year to query
        collection_ids: Optional collection IDs

    Returns:
        int: Number of relevant stories
    """
    start_date, end_date = get_start_end(year)
    if collection_ids:
        results = search_api.story_count(
            query=query,
            start_date=start_date,
            end_date=end_date,
            collection_ids=collection_ids,
        )
    else:
        results = search_api.story_count(
            query=query, start_date=start_date, end_date=end_date, source_ids=source_ids
        )
    return results["relevant"]


def get_mentions_from_source(
    search_api,
    source_ids,
    source_name,
    queries,
    year,
    api_sleep,
    verbose,
    collection_ids=None,
):
    """
    Get mentions of causes of death from a specific source.

    Args:
        config: Configuration object
        search_api: Media Cloud API instance
        source_ids: List of source IDs to query
        source_name: Name of the source
        queries: Dictionary of queries to run
        year: Year to query for
        collection_ids: List of collection IDs to query

    Returns:
        pd.DataFrame: DataFrame containing the results of the queries
    """
    query_count = []
    start_time_overall = time.time()
    for name, query in queries.items():
        time.sleep(
            api_sleep
        )  # Wait to avoid hitting API rate limits - increase sleep if needed
        start_time = time.time()
        cnt = query_results(
            search_api, query, source_ids, collection_ids=collection_ids, year=year
        )
        if verbose:
            time_now = time.time()
            diff_time = time_now - start_time_overall
            minutes_elapsed = int(diff_time / 60)
            secconds_elapsed = int(diff_time - (60 * int(minutes_elapsed)))
            Log.elapsed(minutes_elapsed, secconds_elapsed)
            Log.query(source_name, name, query, cnt, time.time() - start_time)
        query_count.append(
            {
                "cause": name,
                "mentions": cnt,
                "source": source_name,
                "year": year,
            }
        )
    return pd.DataFrame(query_count)


def get_media_mentions(
    outlets: list,
    causes_of_death: list[str],
    rerun_queries: bool,
    api_token: str | None,
    queries: dict,
    collections: list,
    year: int,
    api_sleep: float,
    verbose: bool,
    overwrite: bool,
    run_single_queries: bool,
    output_dir,
):
    """
    Get media mentions from Media Cloud or load cached data.

    Args:
        config: Configuration object
        outlets: List of outlets to query
        causes_of_death: List of causes of death to query
        run_single_queries: Whether to run single keyword queries

    Returns:
        pd.DataFrame: Media mentions data
    """
    if rerun_queries:
        if not api_token:
            raise ValueError(
                "config.MC_API_TOKEN not set. Get API key from https://www.mediacloud.org/ "
                "or set config.RERUN_QUERIES=False to use cached data"
            )

        Log.section("QUERYING MEDIA CLOUD API")
        Log.warning(
            "This may take some minutes due to API rate limits (2 requests per minute)"
        )

        # Initialize search API
        search_api = mediacloud.api.SearchApi(api_token)

        # Create queries
        STR_QUERIES = create_full_queries(queries)
        SINGLE_QUERIES = create_single_keyword_queries(queries)

        queries_in_use = {
            q: q_str for q, q_str in STR_QUERIES.items() if q in causes_of_death
        }
        if run_single_queries:
            SINGLE_QUERIES = create_single_keyword_queries(queries)
            single_queries_in_use = {
                q: q_str for q, q_str in SINGLE_QUERIES.items() if q in causes_of_death
            }
            single_mentions_ls = []

        mentions_ls = []

        # Query each source
        for outlet in outlets:
            s_id = outlet["id"]
            s_name = outlet["full_name"]
            mentions = get_mentions_from_source(
                search_api=search_api,
                source_ids=[s_id],
                source_name=s_name,
                queries=queries_in_use,
                year=year,
                api_sleep=api_sleep,
                verbose=verbose,
            )
            mentions_ls.append(mentions.copy(deep=True))

            if not run_single_queries:
                continue  # Skip single keyword queries if not needed
            single_mentions = get_mentions_from_source(
                search_api=search_api,
                source_ids=[s_id],
                source_name=s_name,
                queries=single_queries_in_use,
                year=year,
                api_sleep=api_sleep,
                verbose=verbose,
            )
            single_mentions_ls.append(single_mentions.copy(deep=True))

        # Add mentions for collections
        for collection in collections:
            c_id = collection["id"]
            c_name = collection["full_name"]
            collection_mentions = get_mentions_from_source(
                search_api=search_api,
                source_ids=[],
                source_name=c_name,
                queries=queries_in_use,
                year=year,
                api_sleep=api_sleep,
                collection_ids=[c_id],
                verbose=verbose,
            )
            mentions_ls.append(collection_mentions.copy(deep=True))

            if not run_single_queries:
                continue  # Skip single keyword queries if not needed
            collection_single_mentions = get_mentions_from_source(
                search_api=search_api,
                source_ids=[],
                source_name=c_name,
                queries=single_queries_in_use,
                year=year,
                api_sleep=api_sleep,
                verbose=verbose,
                collection_ids=[c_id],
            )
            single_mentions_ls.append(collection_single_mentions.copy(deep=True))

        # Concatenate all mentions
        mentions_df = pd.concat(mentions_ls, ignore_index=True)

        if run_single_queries:
            single_mentions_df = pd.concat(single_mentions_ls, ignore_index=True)
            single_mentions_df = single_mentions_df.rename(
                columns={"mentions": "single_mentions"}
            )
            mentions_df = mentions_df.merge(
                single_mentions_df, on=["cause", "source", "year"], how="left"
            )
            mentions_df = mentions_df[
                ["year", "source", "cause", "mentions", "single_mentions"]
            ]

        if overwrite:
            mentions_file = output_dir / "media_deaths_mentions.csv"
            mentions_df.to_csv(mentions_file, index=False)
            Log.success(f"Saved mentions data to {mentions_file}")
    else:
        mentions_file = output_dir / "media_deaths_mentions.csv"
        Log.info(f"Loading cached mentions data from {mentions_file}")
        mentions_df = pd.read_csv(mentions_file)

    mentions_df = mentions_df.astype({"mentions": "Int64", "year": "Int64"})
    return mentions_df


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================


def add_shares(tb, columns=None):
    """
    Add shares for each row relative to total of columns to DataFrame.

    Args:
        tb: DataFrame to add shares to
        columns: List of columns to calculate shares for

    Returns:
        pd.DataFrame: DataFrame with added share columns
    """
    if columns is None:
        columns = ["mentions", "deaths"]

    for col in columns:
        total = tb[col].sum()
        if total == 0:
            tb.loc[:, f"{col}_share"] = 0
        else:
            tb.loc[:, f"{col}_share"] = round((tb[col] / total) * 100, 3)

    return tb


def analyze_data(
    mentions_df,
    death_df,
    causes_of_death: list[str],
    outlets: list,
    collections: list,
    run_single_queries: bool,
):
    """
    Analyze media mentions and deaths data.

    Args:
        config: Configuration object
        mentions_df: DataFrame with media mentions
        death_df: DataFrame with deaths data
        causes_of_death: List of causes of death to analyze
        run_single_queries: Whether single keyword queries were run

    Returns:
        pd.DataFrame: Analyzed and pivoted data
    """
    Log.info("Analyzing data...")

    # Copy dataframes
    tb_mentions = mentions_df.copy(deep=True)
    tb_deaths = death_df.copy(deep=True)

    # Filter only on causes of death we are interested in
    tb_mentions = tb_mentions[tb_mentions["cause"].isin(causes_of_death)]

    # Merge with deaths data
    tb_mentions = pd.merge(
        left=tb_mentions, right=tb_deaths, on=["cause", "year"], how="left"
    )

    sources = tb_mentions["source"].unique().tolist()

    # Add shares to media mentions table
    tb_mentions.loc[:, "mentions_share"] = 0.0
    tb_mentions.loc[:, "deaths_share"] = 0.0
    if run_single_queries:
        tb_mentions.loc[:, "single_mentions_share"] = 0.0

    for source in sources:
        tb_s = tb_mentions[tb_mentions["source"] == source]
        if run_single_queries:
            tb_s = add_shares(tb_s, columns=["mentions", "deaths", "single_mentions"])
        else:
            tb_s = add_shares(tb_s, columns=["mentions", "deaths"])
        tb_mentions.update(tb_s)

    # Pivot table
    if run_single_queries:
        tb_mentions = tb_mentions.pivot(
            index=["cause", "year", "deaths", "deaths_share"],
            columns="source",
            values=[
                "mentions",
                "mentions_share",
                "single_mentions",
                "single_mentions_share",
            ],
        ).reset_index()

    else:
        tb_mentions = tb_mentions.pivot(
            index=["cause", "year", "deaths", "deaths_share"],
            columns="source",
            values=["mentions", "mentions_share"],
        ).reset_index()

    columns_flat = ["cause", "year", "deaths", "deaths_share"]
    short_names = [outlet["short_name"] for outlet in outlets] + [
        collection["short_name"] for collection in collections
    ]
    for metric in ["mentions", "share", "single_mentions", "single_share"]:
        if not run_single_queries and metric in ["single_mentions", "single_share"]:
            continue
        for short_name in short_names:
            columns_flat.append(f"{short_name}_{metric}")

    tb_mentions.columns = columns_flat

    return tb_mentions


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def plot_media_deaths_matplotlib(
    # config: Config,
    media_deaths_df,
    causes_of_death: list[str],
    fixed_colors,
    year,
    columns=None,
    bar_labels=None,
    title=None,
    absolute=False,
    save_path=None,
):
    """
    Plot media deaths data using matplotlib.

    Args:
        config: Configuration object
        media_deaths_df: DataFrame containing media deaths data
        causes_of_death: List of causes of death to plot
        columns: List of columns to plot
        bar_labels: List of labels for the bars
        title: Title of the plot
        absolute: If True, show absolute values instead of percentages
        save_path: Path to save the plot
        fixed_colors: Dictionary of fixed colors for each cause
    """
    if columns is None:
        columns = ["deaths_share", "mentions_share"]
        bar_labels = ["Deaths", "Mentions"]
    if bar_labels is None:
        bar_labels = columns
    if title is None:
        title = f"Media Mentions of Causes of Death in {year}"

    mm_plot = media_deaths_df[["cause"] + columns].transpose()
    mm_plot.columns = mm_plot.iloc[0]
    mm_plot = mm_plot.drop(mm_plot.index[0])
    mm_plot.index = bar_labels
    max_val = mm_plot.sum(axis=1).max()

    ordered_cols = [cause for cause in causes_of_death if cause in mm_plot.columns]
    color_order = [fixed_colors[cause] for cause in ordered_cols]

    mm_plot = mm_plot[ordered_cols]
    ax = mm_plot.plot(kind="bar", stacked=True, color=color_order)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    if absolute:
        plt.ylabel("Count")
    else:
        plt.ylabel("Share")
    plt.title(title, loc="center")
    plt.legend(title="Cause of death", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    for i, row in enumerate(mm_plot.values):
        cumulative = 0
        for j, value in enumerate(row):
            if value > (max_val * 0.02):
                if absolute:
                    seg_label = f"{int(value)}"
                else:
                    seg_label = f"{round(value, 1)}%"
                ax.text(
                    x=i,
                    y=cumulative + value / 2,
                    s=seg_label,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
            cumulative += value

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        Log.success(f"Saved plot to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
