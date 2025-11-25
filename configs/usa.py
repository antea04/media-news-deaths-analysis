"""USA configuration for media deaths analysis."""

# Import queries from existing query_generation module
# (This maintains backward compatibility with the original script)
from query_generation import create_queries

CONFIG = {
    # Country identification
    "country_name": "United States",
    "country_code": "USA",
    "language": "en",

    # Analysis year
    "year": 2023,

    # Death data source
    "death_data_source": "cdc_wonder",
    "death_data_urls": {
        "leading_causes": "https://snapshots.owid.io/6f/b0139e189d66756d94f84fafab7c3c",
        "external_causes": "https://snapshots.owid.io/27/cb223d374b691fbd451c1985d0cf31",
    },

    # Additional data
    "terrorism_deaths": 16,  # From Global Terrorism Index 2023

    # Media outlets (Media Cloud source IDs)
    "outlets": [
        {"full_name": "The New York Times", "id": 1, "short_name": "nyt"},
        {"full_name": "The Washington Post", "id": 2, "short_name": "wapo"},
        {"full_name": "Fox News", "id": 1092, "short_name": "fox"},
    ],

    # Media Cloud collections
    "collections": [
        {"full_name": "US Collection", "id": 34412234, "short_name": "us"}
    ],

    # Causes of death to analyze (standardized names)
    "causes_of_death": [
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
    ],

    # Mapping from data source categories to standardized names
    # For CDC Wonder: maps their category names to our standardized names
    "causes_map": {
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
    },

    # Colors for visualization (can be customized per country)
    "colors": {
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
    },

    # API settings
    "api_sleep": 10,  # Seconds to wait between API calls

    # ========================================================================
    # MEDIA CLOUD QUERY KEYWORDS
    # ========================================================================
    # These keywords are used to search for articles about each cause of death.
    # Each cause has:
    #   - single_terms: Keywords that must appear in the article
    #   - combinations: Pairs of terms for proximity searches (within 1000 words)
    #   - exclude_terms: Terms that indicate false positives to exclude
    #
    # NOTE: Uses the existing query_generation module for backward compatibility
    # ========================================================================
    "queries": create_queries(),  # Import from query_generation.py
}
