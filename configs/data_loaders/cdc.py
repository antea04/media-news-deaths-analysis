import pandas as pd
from media_deaths.data_loaders import register_loader

USER_AGENT = {"User-Agent": "Mozilla/5.0"}
YEAR = 2023
# Terrorism deaths for the USA from the Global Terrorism Index
# If needed, update from here: https://www.visionofhumanity.org/maps/global-terrorism-index/#/
TERRORISM_DEATHS_2023 = 16


@register_loader("cdc")
def load_data():
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
    df = format_death_data(leading_causes_df, external_causes_df)

    # Drop NAs
    df = df.dropna(subset="cause")
    return df


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
