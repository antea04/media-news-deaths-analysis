"""Data loader for Catalan mortality data from GenCat Salut.

This loader fetches mortality data from the Generalitat de Catalunya's health department.
Data source: https://scientiasalut.gencat.cat/handle/11351/13451.2

The loader processes the data to extract causes of death and aggregate them into
standard categories (cancer, heart disease, homicide, etc.).
"""

import pandas as pd
from media_deaths.data_loaders import register_loader


@register_loader("catalan_gencat")
def load_data() -> pd.DataFrame:
    """Load and process mortality data from GenCat Salut.

    Args:
        config: Configuration object (uses config.YEAR)

    Returns:
        DataFrame with columns: ['code', 'cause', 'deaths', 'year']
    """
    print("Loading deaths data from GenCat Salut...")

    # Fetch data
    df = _fetch_data()

    # Select relevant rows and clean
    df = _clean_data(df)

    # Discard columns by sex (keep code column)
    df = df[["code", "cause", "total_deaths"]]
    df = df.rename(columns={"total_deaths": "deaths"})
    df["year"] = 2023

    # Aggregate causes into standard categories
    df = _aggregate_causes(df)

    # Ensure correct data types
    df = df.astype({"deaths": "Int64", "year": "Int64"})

    return df


def _fetch_data() -> pd.DataFrame:
    """Fetch raw data from GenCat Salut Excel file.

    Returns:
        Raw DataFrame from Excel file
    """
    file_url = "https://scientiasalut.gencat.cat/bitstream/handle/11351/13451.2/analisi-mortalitat-catalunya-2023-taules.xlsx?sequence=2&isAllowed=y"
    sheet_name = "73 grups de causes"
    df = pd.read_excel(file_url, sheet_name=sheet_name, skiprows=235)
    return df


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and format raw data from GenCat Salut.

    Steps:
    - Select relevant rows
    - Drop unnecessary columns
    - Rename columns to English
    - Extract cause codes and names
    - Sort by total deaths

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    TRANSLATIONS = {
        "Dones": "female",
        "Homes": "male",
        "Defuncions": "deaths",
        "Percentatge": "share",
        "Total": "total",
    }

    # Select relevant rows
    NUM_ROWS = 75
    df = df.head(NUM_ROWS)

    # Sanity check - verify expected format
    assert "Dones" in df.columns, "Expected 'Dones' column in data"
    assert df.loc[0, "Dones"] == "Defuncions", "Expected 'Defuncions' in first row"
    assert df.loc[1, "Unnamed: 2"] == "1  .Infeccioses intestinals", (
        "Unexpected data format"
    )
    assert df.loc[NUM_ROWS - 1, "Unnamed: 2"] == "Total", "Expected 'Total' in last row"

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

    # Drop first row (header)
    df = df.drop(index=0).reset_index(drop=True)

    # Extract code and cause name from format "[NUMBER] .[CAUSE_NAME]"
    def extract_code_and_cause(text):
        """Extract code number and cause name."""
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

    # Sort by total deaths descending
    df = df.sort_values("total_deaths", ascending=False)

    return df


def _aggregate_causes(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate specific causes into standard categories.

    Maps Catalan cause names to standard categories:
    - cancer: All tumors (malignant and benign)
    - heart disease: Ischemic heart disease, other heart diseases, heart failure
    - homicide: Homicides

    Args:
        df: DataFrame with individual causes

    Returns:
        DataFrame with aggregated causes
    """
    # Define cause mappings
    CAUSES = {
        "heart disease": {
            "causes": [
                "Isquèmiques del cor",
                "Resta del cor",
                "Insuficiència cardíaca",
            ]
        },
        "homicide": {
            "causes": ["Homicidis"],
        },
    }

    # Cancer: all rows containing tumor markers
    mask_cancer = (
        df["cause"].str.contains("T.M.")
        | df["cause"].str.contains("T. benignes")
        | df["cause"].isin(["Limfoma", "Leucèmia"])
    )
    df.loc[mask_cancer, "cause_agg"] = "cancer"

    # Other causes: exact matches
    for cause in CAUSES:
        mask = df["cause"].isin(CAUSES[cause]["causes"])
        df.loc[mask, "cause_agg"] = cause

    # Keep only causes that were mapped
    df = df.dropna(subset=["cause_agg"])

    # Aggregate by cause and year
    df = df.groupby(["cause_agg", "year"], as_index=False)["deaths"].sum()
    df = df.rename(columns={"cause_agg": "cause"})

    return df
