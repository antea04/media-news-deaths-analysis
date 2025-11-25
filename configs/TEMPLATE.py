"""
TEMPLATE for creating a new country configuration.

To add a new country:
1. Copy this file to [country_code].py (e.g., spain.py, brazil.py)
2. Fill in all the values below
3. Create corresponding queries/[language].py with translated keywords
4. Run: python run_analysis.py --country [country_code]

Example: configs/spain.py + queries/es.py
"""

CONFIG = {
    # ============================================================================
    # COUNTRY IDENTIFICATION
    # ============================================================================
    "country_name": "Country Name",  # e.g., "Spain", "Brazil", "France"
    "country_code": "XXX",  # ISO 3-letter code, e.g., "ESP", "BRA", "FRA"
    "language": "xx",  # ISO 2-letter language code, e.g., "es", "pt", "fr"

    # ============================================================================
    # ANALYSIS YEAR
    # ============================================================================
    "year": 2023,  # Year to analyze

    # ============================================================================
    # DEATH DATA SOURCE
    # ============================================================================
    # Options: "cdc_wonder", "who", "custom_url"
    "death_data_source": "who",  # Most countries should use "who"

    # If using CDC Wonder (USA only), provide URLs:
    # "death_data_urls": {
    #     "leading_causes": "https://...",
    #     "external_causes": "https://...",
    # },
    #
    # If using WHO (recommended for non-USA):
    "death_data_urls": None,  # WHO data loaded automatically via OWID catalog

    # ============================================================================
    # ADDITIONAL DATA
    # ============================================================================
    # Terrorism deaths (if not included in main data source)
    # Get from: https://www.visionofhumanity.org/maps/global-terrorism-index/
    "terrorism_deaths": 0,

    # ============================================================================
    # MEDIA OUTLETS
    # ============================================================================
    # Find Media Cloud source IDs at: https://search.mediacloud.org/
    # Search for newspapers and note their ID numbers
    "outlets": [
        # Example:
        # {"full_name": "El País", "id": 12345, "short_name": "elpais"},
        # {"full_name": "Le Monde", "id": 67890, "short_name": "lemonde"},
        # {"full_name": "Folha de S.Paulo", "id": 11111, "short_name": "folha"},
    ],

    # ============================================================================
    # MEDIA CLOUD COLLECTIONS (optional)
    # ============================================================================
    # Find collections at: https://search.mediacloud.org/collections/news/geographic
    "collections": [
        # Example:
        # {"full_name": "Spain Collection", "id": 99999, "short_name": "spain"}
    ],

    # ============================================================================
    # CAUSES OF DEATH TO ANALYZE
    # ============================================================================
    # Keep these standardized names - they map to your translated queries
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

    # ============================================================================
    # DEATH CAUSE MAPPING
    # ============================================================================
    # Maps data source category names to standardized names
    #
    # For CDC Wonder (USA):
    # "causes_map": {
    #     "#Diseases of heart (I00-I09,I11,I13,I20-I51)": "heart disease",
    #     "#Malignant neoplasms (C00-C97)": "cancer",
    #     ...
    # }
    #
    # For WHO (most countries):
    "causes_map": {},  # WHO data doesn't need mapping (handled by load_who_data.py)

    # ============================================================================
    # VISUALIZATION COLORS
    # ============================================================================
    # Colors for each cause (hex codes)
    "colors": {
        "heart disease": "#1f77b4",
        "cancer": "#ff7f0e",
        "accidents": "#2ca02c",
        "stroke": "#d62728",
        "respiratory": "#9467bd",
        "alzheimers": "#8c564b",
        "diabetes": "#e377c2",
        "kidney": "#7f7f7f",
        "liver": "#bcbd22",
        "covid": "#17becf",
        "suicide": "#aec7e8",
        "influenza": "#ffbb78",
        "drug overdose": "#98df8a",
        "homicide": "#ff9896",
        "terrorism": "#c5b0d5",
        "war": "#c49c94",
        "hiv": "#f7b6d2",
        "malaria": "#c7c7c7",
        "tb": "#dbdb8d",
        "diarrhea": "#9edae5",
    },

    # ============================================================================
    # API SETTINGS
    # ============================================================================
    "api_sleep": 10,  # Seconds to wait between Media Cloud API calls
}
