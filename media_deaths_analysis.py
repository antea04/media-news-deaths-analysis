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
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import mediacloud.api
from dotenv import load_dotenv

from query_generation import (
    create_full_queries,
    create_single_keyword_queries,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Set overall variables for analysis
YEAR = 2023
VERBOSE = True
LANGUAGE = "en"

# Load API token from environment variables
load_dotenv()
MC_API_TOKEN = os.getenv("MC_API_TOKEN")
API_SLEEP = 10

USER_AGENT = {"User-Agent": "Mozilla/5.0"}

# Terrorism deaths for the USA from the Global Terrorism Index
# If needed, update from here: https://www.visionofhumanity.org/maps/global-terrorism-index/#/
TERRORISM_DEATHS_2023 = 16

# Whether you want to rerun the queries or not - rerunning the queries can take ~30 minutes.
# If false, the script uses the results found in ./data/media_deaths_mentions.csv
RERUN_QUERIES = True

# Whether you want to run single keyword queries (only 1 mention per article) vs multiple keyword queries (multiple mentions per article)
RUN_SINGLE_QUERIES = False

# Whether to use final results for all plots. If TRUE this overrides all analysis
# and uses the saved results in ./data/media_deaths_results.csv
USE_SAVED_RESULTS = False
# Whether you want to overwrite existing files or not
OVERWRITE = True


# Causes of death we are using for the 2023 version
# Based on the 12 leading causes of death in the US for 2023,
# plus drug overdoses, homicides, and terrorism
CAUSES_OF_DEATH = [
    "heart disease",
    "cancer",
    "accidents",
    "stroke",
    "respiratory",
    "alzheimers",
    "diabetes",
    "kidney",
    "liver",
    "covid",
    "suicide",
    "influenza",
    "drug overdose",
    "homicide",
    "terrorism",
]

# Colors used for charts
FIXED_COLOURS = {
    "heart disease": "#1f77b4",  # Blue
    "cancer": "#ff7f0e",  # Orange
    "accidents": "#2ca02c",  # Green
    "stroke": "#d62728",  # Red
    "respiratory": "#9467bd",  # Purple
    "alzheimers": "#8c564b",  # Brown
    "diabetes": "#e377c2",  # Pink
    "kidney": "#7f7f7f",  # Gray
    "liver": "#bcbd22",  # Olive
    "covid": "#17becf",  # Teal
    "suicide": "#aec7e8",  # Light blue
    "influenza": "#ffbb78",  # Light orange
    "drug overdose": "#98df8a",  # Light green
    "homicide": "#ff9896",  # Light red
    "terrorism": "#c5b0d5",  # Light purple
    "war": "#c49c94",  # Light brown
    "hiv": "#f7b6d2",  # Light pink
    "malaria": "#c7c7c7",  # Light gray
    "tb": "#dbdb8d",  # Light olive
    "diarrhea": "#9edae5",  # Light teal
}

# Media Cloud source IDs
NYT_ID = 1
WAPO_ID = 2
FOX_ID = 1092
US_COLLECTION_ID = 34412234

# Media outlets information, replace this with any other outlets if needed
OUTLETS = [
    {"full_name": "The New York Times", "id": 1, "short_name": "nyt"},
    {"full_name": "The Washington Post", "id": 2, "short_name": "wapo"},
    {"full_name": "Fox News", "id": 1092, "short_name": "fox"},
]
# collections information, replace or add other collections if needed
COLLECTIONS = [{"full_name": "US Collection", "id": 34412234, "short_name": "us"}]


# ===============================================================
# DATA PROCESSING FUNCTIONS
# ==============================================================


def format_death_data(leading_causes_df, external_causes_df):
    """
    Format/process deaths data from CDC Wonder database.
    Replace with specific death file for country if needed.

    Returns:
        pd.DataFrame: Processed deaths data with columns: cause, year, deaths
    """

    # Map CDC names to our keywords
    CAUSES_MAP = {
        "#Diseases of heart (I00-I09,I11,I13,I20-I51)": "heart disease",
        "#Malignant neoplasms (C00-C97)": "cancer",
        "#Accidents (unintentional injuries) (V01-X59,Y85-Y86)": "accidents",
        "#Cerebrovascular diseases (I60-I69)": "stroke",
        "#Chronic lower respiratory diseases (J40-J47)": "respiratory",
        "#Alzheimer disease (G30)": "alzheimers",
        "#Diabetes mellitus (E10-E14)": "diabetes",
        "#Nephritis, nephrotic syndrome and nephrosis (N00-N07,N17-N19,N25-N27)": "kidney",
        "#Chronic liver disease and cirrhosis (K70,K73-K74)": "liver",
        "#COVID-19 (U07.1)": "covid",
        "#Intentional self-harm (suicide) (*U03,X60-X84,Y87.0)": "suicide",
        "#Influenza and pneumonia (J09-J18)": "influenza",
    }

    # Process leading causes
    leading_causes_df["cause"] = leading_causes_df["15 Leading Causes of Death"].map(
        CAUSES_MAP
    )
    leading_causes_df = leading_causes_df.drop(
        columns=[
            "Notes",
            "Population",
            "15 Leading Causes of Death",
            "15 Leading Causes of Death Code",
            "Crude Rate",
        ],
        errors="raise",
    )
    leading_causes_df = leading_causes_df.dropna(subset=["cause", "Deaths"], how="all")
    leading_causes_df["year"] = YEAR

    # Format external causes df
    # Replace Suppressed/Unreliable with pd.NA
    external_causes_df = external_causes_df.replace("Suppressed", pd.NA)
    external_causes_df = external_causes_df.replace("Unreliable", pd.NA)
    external_causes_df["Deaths"] = external_causes_df["Deaths"].astype("Int64")
    external_causes_df = external_causes_df.drop(
        columns=["Notes", "Population", "ICD Sub-Chapter Code"], errors="raise"
    )
    external_causes_df["year"] = YEAR

    # Combine both dataframes and add terrorism deaths
    death_df = create_tb_death(leading_causes_df, external_causes_df)

    print(f"Loaded death data for {len(death_df)} causes")
    return death_df


def create_tb_death(tb_leading_causes, tb_ext_causes):
    """
    Combine leading causes and external causes data.

    Args:
        tb_leading_causes: DataFrame with leading causes
        tb_ext_causes: DataFrame with external causes

    Returns:
        pd.DataFrame: Combined deaths data
    """
    # Get drug overdose deaths
    drug_od_deaths = tb_ext_causes[tb_ext_causes["Cause of death Code"] == "X42"][
        "Deaths"
    ].iloc[0]

    # Get homicide deaths
    ext_causes_gb = (
        tb_ext_causes[["Deaths", "ICD Sub-Chapter"]]
        .groupby("ICD Sub-Chapter")
        .sum()
        .reset_index()
    )
    homicide_deaths = ext_causes_gb[ext_causes_gb["ICD Sub-Chapter"] == "Assault"][
        "Deaths"
    ].iloc[0]

    terrorism_deaths = TERRORISM_DEATHS_2023

    deaths = [
        {"cause": "drug overdose", "year": YEAR, "deaths": drug_od_deaths},
        {"cause": "homicide", "year": YEAR, "deaths": homicide_deaths},
        {"cause": "terrorism", "year": YEAR, "deaths": terrorism_deaths},
    ]

    tb_leading_causes.columns = [col.lower() for col in tb_leading_causes.columns]
    tb_deaths = pd.concat([tb_leading_causes, pd.DataFrame(deaths)])

    # Subtract drug overdose deaths from accidents
    acc_deaths = tb_deaths[tb_deaths["cause"] == "accidents"]["deaths"].iloc[0]
    drug_od_deaths = tb_deaths[tb_deaths["cause"] == "drug overdose"]["deaths"].iloc[0]
    tb_deaths.loc[tb_deaths["cause"] == "accidents", "deaths"] = (
        acc_deaths - drug_od_deaths
    )

    return tb_deaths


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
    year=YEAR,
    collection_ids=None,
):
    """
    Get mentions of causes of death from a specific source.

    Args:
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
            API_SLEEP
        )  # Wait to avoid hitting API rate limits - increase sleep if needed
        start_time = time.time()
        cnt = query_results(
            search_api, query, source_ids, collection_ids=collection_ids, year=year
        )
        if VERBOSE:
            time_now = time.time()
            diff_time = time_now - start_time_overall
            minutes_elapsed = int(diff_time / 60)
            secconds_elapsed = int(diff_time - (60 * int(minutes_elapsed)))
            print(f"Time elapsed: {minutes_elapsed:02d}:{secconds_elapsed:02d} minutes")
            print(f"Querying: {source_name} for CoD {name}")
            print(f"Query: {query}")
            print(
                f"Count: {cnt} mentions for {name} in the {source_name} "
                f"in {year} - retrieved in {time.time() - start_time:.2f} seconds"
            )
            print("-" * 40)
        query_count.append(
            {
                "cause": name,
                "mentions": cnt,
                "source": source_name,
                "year": year,
            }
        )
    return pd.DataFrame(query_count)


def get_media_mentions(outlets=OUTLETS, run_single_queries=RUN_SINGLE_QUERIES):
    """
    Get media mentions from Media Cloud or load cached data.

    Returns:
        pd.DataFrame: Media mentions data
    """
    if RERUN_QUERIES:
        if not MC_API_TOKEN:
            raise ValueError(
                "MC_API_TOKEN not set. Get API key from https://www.mediacloud.org/ "
                "or set RERUN_QUERIES=False to use cached data"
            )

        print("Querying Media Cloud API...")
        print(
            "This may take ~30 minutes due to API rate limits (2 requests per minute)..."
        )

        # Initialize search API
        search_api = mediacloud.api.SearchApi(MC_API_TOKEN)

        # Create queries
        STR_QUERIES = create_full_queries()
        SINGLE_QUERIES = create_single_keyword_queries()

        queries_in_use = {
            q: q_str for q, q_str in STR_QUERIES.items() if q in CAUSES_OF_DEATH
        }
        if run_single_queries:
            SINGLE_QUERIES = create_single_keyword_queries()
            single_queries_in_use = {
                q: q_str for q, q_str in SINGLE_QUERIES.items() if q in CAUSES_OF_DEATH
            }
            single_mentions_ls = []

        mentions_ls = []

        # Query each source
        for outlet in outlets:
            s_id = outlet["id"]
            s_name = outlet["full_name"]
            mentions = get_mentions_from_source(
                search_api, [s_id], s_name, queries_in_use, year=YEAR
            )
            mentions_ls.append(mentions.copy(deep=True))

            if not run_single_queries:
                continue  # Skip single keyword queries if not needed
            single_mentions = get_mentions_from_source(
                search_api, [s_id], s_name, single_queries_in_use, year=YEAR
            )
            single_mentions_ls.append(single_mentions.copy(deep=True))

        # Add mentions for collections
        for collection in COLLECTIONS:
            c_id = collection["id"]
            c_name = collection["full_name"]
            collection_mentions = get_mentions_from_source(
                search_api,
                source_ids=[],
                source_name=c_name,
                queries=queries_in_use,
                year=YEAR,
                collection_ids=[c_id],
            )
            mentions_ls.append(collection_mentions.copy(deep=True))

            if not run_single_queries:
                continue  # Skip single keyword queries if not needed
            collection_single_mentions = get_mentions_from_source(
                search_api,
                source_ids=[],
                source_name=c_name,
                queries=single_queries_in_use,
                year=YEAR,
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

        if OVERWRITE:
            os.makedirs("./data", exist_ok=True)
            mentions_df.to_csv(
                f"./data/media_deaths_mentions_{LANGUAGE}.csv", index=False
            )
            print(f"Saved mentions data to ./data/media_deaths_mentions_{LANGUAGE}.csv")
    else:
        print(
            f"Loading cached mentions data from ./data/media_deaths_mentions_{LANGUAGE}.csv..."
        )
        mentions_df = pd.read_csv(f"./data/media_deaths_mentions_{LANGUAGE}.csv")
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


def analyze_data(mentions_df, death_df, run_single_queries=RUN_SINGLE_QUERIES):
    """
    Analyze media mentions and deaths data.

    Args:
        mentions_df: DataFrame with media mentions
        death_df: DataFrame with deaths data

    Returns:
        pd.DataFrame: Analyzed and pivoted data
    """
    print("Analyzing data...")

    # Copy dataframes
    tb_mentions = mentions_df.copy(deep=True)
    tb_deaths = death_df.copy(deep=True)

    # Filter only on causes of death we are interested in
    tb_mentions = tb_mentions[tb_mentions["cause"].isin(CAUSES_OF_DEATH)]

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
    short_names = [outlet["short_name"] for outlet in OUTLETS] + [
        collection["short_name"] for collection in COLLECTIONS
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
    media_deaths_df,
    columns=None,
    bar_labels=None,
    title=None,
    absolute=False,
    save_path=None,
    fixed_colors=None,
):
    """
    Plot media deaths data using matplotlib.

    Args:
        media_deaths_df: DataFrame containing media deaths data
        columns: List of columns to plot
        bar_labels: List of labels for the bars
        title: Title of the plot
        absolute: If True, show absolute values instead of percentages
        save_path: Path to save the plot
        fixed_colors: Dictionary of fixed colors for each cause
    """
    if fixed_colors is None:
        fixed_colors = FIXED_COLOURS

    if columns is None:
        columns = ["deaths_share", "mentions_share"]
        bar_labels = ["Deaths", "Mentions"]
    if bar_labels is None:
        bar_labels = columns
    if title is None:
        title = f"Media Mentions of Causes of Death in {YEAR}"

    mm_plot = media_deaths_df[["cause"] + columns].transpose()
    mm_plot.columns = mm_plot.iloc[0]
    mm_plot = mm_plot.drop(mm_plot.index[0])
    mm_plot.index = bar_labels
    max_val = mm_plot.sum(axis=1).max()

    ordered_cols = [cause for cause in CAUSES_OF_DEATH if cause in mm_plot.columns]
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
        print(f"Saved plot to {save_path}")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main(outlets=OUTLETS):
    """Main execution function."""
    print("=" * 80)
    print("MEDIA DEATHS ANALYSIS")
    print("=" * 80)
    print(f"Year: {YEAR}")
    print(f"Language: {LANGUAGE}")
    print(f"Outlets: {[outlet['full_name'] for outlet in OUTLETS]}")
    print(f"Rerun queries: {RERUN_QUERIES}")
    print(f"Run single keyword queries: {RUN_SINGLE_QUERIES}")
    print(f"Use saved results: {USE_SAVED_RESULTS}")
    print("=" * 80)
    print()

    # Load or use saved results
    if USE_SAVED_RESULTS:
        print(
            f"Loading saved results from ./data/media_deaths_results_{LANGUAGE}.csv..."
        )
        media_deaths_df = pd.read_csv(f"./data/media_deaths_results_{LANGUAGE}.csv")
    else:
        # Load death data from CDC snapshots
        print("Loading deaths data...")
        leading_causes_df = pd.read_csv(
            "https://snapshots.owid.io/6f/b0139e189d66756d94f84fafab7c3c",
            sep="\t",
            storage_options=USER_AGENT,
        )
        external_causes_df = pd.read_csv(
            "https://snapshots.owid.io/27/cb223d374b691fbd451c1985d0cf31",
            storage_options=USER_AGENT,
        )
        # format death data
        death_df = format_death_data(leading_causes_df, external_causes_df)
        print()

        # Get media mentions
        mentions_df = get_media_mentions(OUTLETS, run_single_queries=RUN_SINGLE_QUERIES)
        print()

        # Analyze data
        media_deaths_df = analyze_data(mentions_df, death_df, RUN_SINGLE_QUERIES)

        # Save results
        if OVERWRITE:
            os.makedirs("./data", exist_ok=True)
            media_deaths_df.to_csv(
                f"./data/media_deaths_results_{LANGUAGE}.csv", index=False
            )
            print(
                f"Saved analysis results to ./data/media_deaths_results_{LANGUAGE}.csv"
            )
        print()

    # Display summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(
        media_deaths_df[
            [
                "cause",
                "deaths",
                "nyt_mentions",
                "wapo_mentions",
                "fox_mentions",
                "us_mentions",
            ]
        ].to_string(index=False)
    )
    print()

    # Create visualizations
    print("=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    print()

    # 1. Media mentions by source
    print("1. Media mentions by source...")
    plot_media_deaths_matplotlib(
        media_deaths_df,
        columns=["deaths_share", "nyt_share", "wapo_share", "fox_share", "us_share"],
        bar_labels=["Deaths", "NYT", "WaPo", "Fox", "US Collection"],
        absolute=False,
        title=f"Media mentions of causes of death in {YEAR}",
        save_path="data/media_deaths_by_source.png",
    )
    print()

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
