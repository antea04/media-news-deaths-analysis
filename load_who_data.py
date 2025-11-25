#!/usr/bin/env python3
"""
Load death statistics from WHO Mortality Database via OWID catalog.

This module provides functions to fetch and format WHO death data for the media
deaths analysis, supporting multi-country comparisons.

Note: WHO mortality database has broader cause categories than CDC Wonder,
so some causes are approximated or aggregated.
"""

import pandas as pd
from owid.catalog import find_latest


# WHO cause name mapping to standardized cause names
# Note: WHO has broader categories, so some mappings are approximations
WHO_CAUSE_MAPPING = {
    "Cardiovascular diseases": "heart disease",
    "Malignant neoplasms": "cancer",
    "Unintentional injuries": "accidents",
    # "Stroke": "stroke",  # Not separate in WHO broad categories
    "Respiratory diseases": "respiratory",
    # "Alzheimer's disease": "alzheimers",  # Part of Neuropsychiatric conditions
    "Diabetes mellitus, blood and endocrine disorders": "diabetes",
    "Genitourinary diseases": "kidney",  # Includes kidney diseases
    "Digestive diseases": "liver",  # Includes liver diseases
    # "COVID-19": "covid",  # May be in Respiratory infections for 2020
    "Intentional injuries": "suicide",  # Includes suicide and homicide
    "Respiratory infections": "influenza",  # Includes pneumonia/flu
    # Drug overdose not separately categorized in WHO
    # Homicide not separately categorized (part of Intentional injuries)
}


def load_who_deaths(
    country_code: str,
    year: int,
    causes_of_death: list,
    terrorism_deaths: int = 0,
) -> pd.DataFrame:
    """
    Load death statistics from WHO Mortality Database via OWID catalog.

    Args:
        country_code: Three-letter country code (e.g., "BRA", "USA")
        year: Year to fetch data for
        causes_of_death: List of standardized cause names to fetch
        terrorism_deaths: Number of terrorism deaths to add (not in WHO)

    Returns:
        pd.DataFrame with columns: cause, year, deaths

    Note: WHO has broader cause categories than CDC Wonder, so some causes
    may not be available or may be approximations (e.g., stroke is part of
    cardiovascular diseases, alzheimers is part of neuropsychiatric conditions).
    """
    print(f"Loading WHO mortality data for {country_code} in {year}...")

    # Load the WHO mortality database from OWID catalog
    tb = find_latest(
        namespace="who",
        dataset="mortality_database",
        table="mortality_database"
    )

    # Filter for the country and year
    # WHO uses full country names, so we need to map the code
    country_name = get_country_name(country_code)

    # Reset index to access filter columns
    tb = tb.reset_index()

    # Filter for the specific country, year, age, and sex
    tb_filtered = tb[
        (tb["country"] == country_name)
        & (tb["year"] == year)
        & (tb["age_group"] == "all ages")
        & (tb["sex"] == "Both sexes")
    ].copy()

    if len(tb_filtered) == 0:
        raise ValueError(
            f"No WHO data found for {country_name} ({country_code}) in {year}. "
            f"Try a different year or check country name mapping. "
            f"Available years: check WHO mortality database."
        )

    # Create mapping dictionary for the causes we need
    death_data = []

    for std_cause in causes_of_death:
        if std_cause == "terrorism":
            # Terrorism not in WHO, use provided value
            death_data.append({
                "cause": "terrorism",
                "year": year,
                "deaths": terrorism_deaths,
            })
            continue

        # Find the WHO cause name for this standardized cause
        who_cause = None
        for who_name, std_name in WHO_CAUSE_MAPPING.items():
            if std_name == std_cause:
                who_cause = who_name
                break

        if who_cause is None:
            print(f"Warning: No WHO mapping found for '{std_cause}', skipping.")
            continue

        # Find the deaths value for this cause
        cause_data = tb_filtered[tb_filtered["cause"] == who_cause]

        if len(cause_data) == 0:
            print(f"Warning: No data found for cause '{who_cause}' (std: '{std_cause}')")
            continue

        deaths_value = cause_data["number"].iloc[0]

        death_data.append({
            "cause": std_cause,
            "year": year,
            "deaths": int(deaths_value),
        })

    death_df = pd.DataFrame(death_data)

    print(f"Loaded {len(death_df)} causes from WHO mortality database")
    return death_df


def get_country_name(country_code: str) -> str:
    """
    Map three-letter country code to full country name used in OWID data.

    Args:
        country_code: Three-letter ISO code (e.g., "BRA", "USA")

    Returns:
        str: Full country name (e.g., "Brazil", "United States")
    """
    # Common country code mappings
    country_mapping = {
        "USA": "United States",
        "BRA": "Brazil",
        "GBR": "United Kingdom",
        "DEU": "Germany",
        "FRA": "France",
        "ITA": "Italy",
        "ESP": "Spain",
        "CAN": "Canada",
        "AUS": "Australia",
        "JPN": "Japan",
        "CHN": "China",
        "IND": "India",
        "MEX": "Mexico",
        "ARG": "Argentina",
        "CHL": "Chile",
        "COL": "Colombia",
        "PER": "Peru",
        "ZAF": "South Africa",
        "NGA": "Nigeria",
        "EGY": "Egypt",
        "KEN": "Kenya",
        "GHA": "Ghana",
        "RUS": "Russia",
        "POL": "Poland",
        "UKR": "Ukraine",
        "TUR": "Turkey",
        "SAU": "Saudi Arabia",
        "IRN": "Iran",
        "PAK": "Pakistan",
        "BGD": "Bangladesh",
        "IDN": "Indonesia",
        "THA": "Thailand",
        "VNM": "Vietnam",
        "PHL": "Philippines",
        "KOR": "South Korea",
        "MYS": "Malaysia",
        "SGP": "Singapore",
        "NZL": "New Zealand",
        "NOR": "Norway",
        "SWE": "Sweden",
        "DNK": "Denmark",
        "FIN": "Finland",
        "BEL": "Belgium",
        "NLD": "Netherlands",
        "CHE": "Switzerland",
        "AUT": "Austria",
        "PRT": "Portugal",
        "GRC": "Greece",
        "CZE": "Czechia",
        "HUN": "Hungary",
        "ROU": "Romania",
        "BGR": "Bulgaria",
        "HRV": "Croatia",
        "SRB": "Serbia",
        "SVK": "Slovakia",
        "SVN": "Slovenia",
        "LTU": "Lithuania",
        "LVA": "Latvia",
        "EST": "Estonia",
        "IRL": "Ireland",
        "ISL": "Iceland",
    }

    if country_code.upper() in country_mapping:
        return country_mapping[country_code.upper()]
    else:
        raise ValueError(
            f"Unknown country code '{country_code}'. "
            f"Please add mapping to get_country_name() function in load_who_data.py"
        )
