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
from dotenv import load_dotenv

from query_generation import (
    create_full_queries,
    create_single_keyword_queries,
)

MEDIA_OUTLET = [
    "vilaweb.cat",
    "ara.cat",
    "elperiodico.cat",
    "Notícies - 324",  # public
]

#
# Mapping
# code, name_source, name_standard
MAPPING = [
    {
        "code": "A00-A09",
        "name_source": "Infectious and parasitic diseases",
        "name_standard": "Intestinal infectious diseases",
    }
]
# ============================================================================
# LOGGING COLORS
# ============================================================================


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Regular colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


def log_info(msg):
    """Log informational message in blue."""
    print(f"{Colors.BLUE}ℹ {msg}{Colors.RESET}")


def log_success(msg):
    """Log success message in green."""
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")


def log_warning(msg):
    """Log warning message in yellow."""
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.RESET}")


def log_error(msg):
    """Log error message in red."""
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")


def log_query(source, cause, query, count, elapsed):
    """Log query details with structured formatting."""
    print(f"{Colors.CYAN}→ {source}{Colors.RESET} | {Colors.BOLD}{cause}{Colors.RESET}")
    print(f"  {Colors.DIM}Query: {query}{Colors.RESET}")
    print(
        f"  {Colors.GREEN}Found {count:,} mentions{Colors.RESET} {Colors.DIM}({elapsed:.2f}s){Colors.RESET}"
    )


def log_section(title):
    """Log section header."""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}{title}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")


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

# Media outlets information, replace this with any other outlets if needed
OUTLETS = [
    {"full_name": "The New York Times", "id": 1, "short_name": "nyt"},
    {"full_name": "The Washington Post", "id": 2, "short_name": "wapo"},
    {"full_name": "Fox News", "id": 1092, "short_name": "fox"},
]
# collections information, replace or add other collections if needed
COLLECTIONS = [
    {"full_name": "US Collection", "id": 34412234, "short_name": "us"},
]


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
            print(
                f"{Colors.DIM}Elapsed: {minutes_elapsed:02d}:{secconds_elapsed:02d}{Colors.RESET}"
            )
            log_query(source_name, name, query, cnt, time.time() - start_time)
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

        log_section("QUERYING MEDIA CLOUD API")
        log_warning(
            "This may take ~30 minutes due to API rate limits (2 requests per minute)"
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
            log_success(
                f"Saved mentions data to ./data/media_deaths_mentions_{LANGUAGE}.csv"
            )
    else:
        log_info(
            f"Loading cached mentions data from ./data/media_deaths_mentions_{LANGUAGE}.csv"
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
    log_info("Analyzing data...")

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
        log_success(f"Saved plot to {save_path}")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================


TRANSLATIONS = {
    "Dones": "female",
    "Homes": "male",
    "Defuncions": "deaths",
    "Percentatge": "share",
    "Total": "total",
}


def _fetch_data():
    """Load data from GenCat Salut.

    This data belongs to 2023 report on mortality in Catalonia.

    More info: https://scientiasalut.gencat.cat/handle/11351/13451.2
    """
    file_url = "https://scientiasalut.gencat.cat/bitstream/handle/11351/13451.2/analisi-mortalitat-catalunya-2023-taules.xlsx?sequence=2&isAllowed=y"
    sheet_name = "73 grups de causes"
    df = pd.read_excel(file_url, sheet_name=sheet_name, skiprows=235)
    return df


def _clean_data(df):
    """Clean data from GenCat Salut.

    - Select relevant rows
    - Drop unnecessary columns
    - Rename columns
    """
    # Select relevant rows
    NUM_ROWS = 75
    df = df.head(NUM_ROWS)

    # Sanity check
    assert "Dones" in df.columns
    assert df.loc[0, "Dones"] == "Defuncions"
    assert df.loc[1, "Unnamed: 2"] == "1  .Infeccioses intestinals"
    assert df.loc[NUM_ROWS - 1, "Unnamed: 2"] == "Total"

    # Drop unnecessary columns
    df = df.dropna(how="all", axis=1)

    # Rename columns
    ## Sex
    columns_sex = [pd.NA if "Unnamed" in x else x for x in df.columns]
    columns_sex = pd.Series(columns_sex).ffill().tolist()
    columns_sex = [TRANSLATIONS.get(x, x) for x in columns_sex]
    ## Metric
    columns_metric = df.loc[0].to_list()
    columns_metric = [TRANSLATIONS.get(x, x) for x in columns_metric]
    ## Combine
    columns = ["cause"] + [
        f"{s}_{m}" for s, m in zip(columns_sex[1:], columns_metric[1:])
    ]

    df.columns = columns

    # Drop first row
    df = df.drop(index=0).reset_index(drop=True)

    # Extract code and cause name
    def extract_code_and_cause(text):
        """Extract code number and cause name from format '[NUMBER] .[CAUSE_NAME]'."""
        if pd.isna(text) or text == "Total":
            return None, text

        # Split by first occurrence of '.'
        parts = text.split(".", 1)
        if len(parts) == 2:
            code = parts[0].strip()
            cause = parts[1].strip()
            return code, cause
        return None, text

    # Apply extraction
    df[["code", "cause"]] = df["cause"].apply(
        lambda x: pd.Series(extract_code_and_cause(x))
    )
    df["code"] = df["code"].astype("Int64")

    # Sort
    df = df.sort_values("total_deaths", ascending=False)

    return df


def load_leading_causes():
    """Load data from GenCat Salut.

    This data belongs to 2023 report on mortality in Catalonia.

    More info: https://scientiasalut.gencat.cat/handle/11351/13451.2
    """
    # Fetch data
    df = _fetch_data()

    # Select relevant rows
    df = _clean_data(df)

    # Discard columns by sex (keep code column)
    df = df[["code", "cause", "total_deaths"]]
    df = df.rename(columns={"total_deaths": "deaths"})
    df["year"] = YEAR

    # Top 12 causes
    causes_top12 = df[df["cause"] != "Total"]["cause"].to_list()[:12]

    # Mapping
    mapping = {
        "Isquèmiques del cor": {
            "english": "Ischemic heart diseases",
            "keywords": [],
            "short_name": "",
        },
        "Demències": {
            "english": "Dementia",
            "keywords": [],
            "short_name": "demencies",
        },
        "Resta del cor": {
            "english": "Other forms of heart disease",
            "keywords": [],
            "short_name": "",
        },
        "T.M.pulmó": {
            "english": "Malignant neoplasm of bronchus and lung",
            "keywords": [],
            "short_name": "",
        },
        "Insuficiència cardíaca": {
            "english": "Heart failure",
            "keywords": [],
            "short_name": "",
        },
        "Cerebrovasculars": {
            "english": "Cerebrovascular diseases",
            "keywords": [],
            "short_name": "",
        },
        "Resta respiratòries": {
            "english": "Other respiratory diseases",
            "keywords": [],
            "short_name": "",
        },
        "Ronyó": {
            "english": "Kidney",
            "keywords": [],
            "short_name": "",
        },
        "Resta digestiu": {
            "english": "Other diseases of the digestive system",
            "keywords": [],
            "short_name": "",
        },
        "Hipertensives": {
            "english": "Hipertension",
            "keywords": [],
            "short_name": "",
        },
        "Alzheimer": {
            "english": "Alzheimer's",
            "keywords": [],
            "short_name": "",
        },
        "Bronquitis i asma": {
            "english": "Bronchitis and asthma",
            "keywords": [],
            "short_name": "",
        },
        # Extra: homicides, drug abuse, terrorism
        "Homicidis": {
            "english": "Homicides",
            "keywords": [],
            "short_name": "homicide",
        },
        "Terrorisme": {
            "english": "Terrorism",
            "keywords": [],
        },
        "Sobredosi": {
            "english": "Drug overdose",
            "keywords": [],
            "short_name": "drug overdose",
        },
    }

    # Sobredosi: https://govern.cat/salapremsa/notes-premsa/634722/l-any-2023-catalunya-va-evitar-la-mort-en-146-sobredosis-gracies-al-les-seves-politiques-en-danys-i-prevencio

    df = df[df[""]]
    return df


def main(outlets=OUTLETS):
    """Main execution function."""
    print()
    log_section("MEDIA DEATHS ANALYSIS")
    print(f"{Colors.DIM}Year:{Colors.RESET} {YEAR}")
    print(f"{Colors.DIM}Language:{Colors.RESET} {LANGUAGE}")
    print(
        f"{Colors.DIM}Outlets:{Colors.RESET} {', '.join([outlet['full_name'] for outlet in OUTLETS])}"
    )
    print(f"{Colors.DIM}Rerun queries:{Colors.RESET} {RERUN_QUERIES}")
    print(f"{Colors.DIM}Run single keyword queries:{Colors.RESET} {RUN_SINGLE_QUERIES}")
    print(f"{Colors.DIM}Use saved results:{Colors.RESET} {USE_SAVED_RESULTS}")
    print()

    # Load or use saved results
    if USE_SAVED_RESULTS:
        log_info(
            f"Loading saved results from ./data/media_deaths_results_{LANGUAGE}.csv"
        )
        media_deaths_df = pd.read_csv(f"./data/media_deaths_results_{LANGUAGE}.csv")
    else:
        # Load death data from CDC snapshots
        log_info("Loading deaths data from CDC...")
        # format death data
        death_df = load_leading_causes()
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
            log_success(
                f"Saved analysis results to ./data/media_deaths_results_{LANGUAGE}.csv"
            )
        print()

    # Display summary statistics
    log_section("SUMMARY STATISTICS")
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
    log_section("CREATING VISUALIZATIONS")

    # 1. Media mentions by source
    log_info("Generating media mentions by source plot...")
    plot_media_deaths_matplotlib(
        media_deaths_df,
        columns=["deaths_share", "nyt_share", "wapo_share", "fox_share", "us_share"],
        bar_labels=["Deaths", "NYT", "WaPo", "Fox", "US Collection"],
        absolute=False,
        title=f"Media mentions of causes of death in {YEAR}",
        save_path="data/media_deaths_by_source.png",
    )
    print()

    log_section("ANALYSIS COMPLETE")
    log_success("All tasks completed successfully")


# if __name__ == "__main__":
#     main()
