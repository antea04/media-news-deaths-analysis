"""Brazil configuration for media deaths analysis."""

CONFIG = {
    # Country identification
    "country_name": "Brazil",
    "country_code": "BRA",
    "language": "pt",

    # Analysis year
    "year": 2020,  # Using 2020 as WHO data is available for this year

    # Death data source - Use WHO Mortality Database
    "death_data_source": "who",
    "death_data_urls": None,  # WHO data loaded via OWID catalog

    # Additional data
    "terrorism_deaths": 0,  # Very low in Brazil

    # Media outlets (Media Cloud source IDs)
    # Note: Only O Globo has data available for 2020 in Media Cloud
    # Other major Brazilian outlets (Folha, iG, Estadão, UOL, G1) return 0 results or errors for 2020
    "outlets": [
        {"full_name": "O Globo", "id": 83352, "short_name": "globo"},
    ],

    # Media Cloud collections (if available)
    "collections": [],

    # Causes of death to analyze (standardized names)
    # Note: WHO has broad categories, so some causes are not available:
    # - stroke: included in "Cardiovascular diseases"
    # - alzheimers: included in "Neuropsychiatric conditions"
    # - covid: included in "Respiratory infections" (not separate in 2020 WHO data)
    # - drug overdose: not separately categorized
    # - homicide: included in "Intentional injuries"
    "causes_of_death": [
        "heart disease",
        "cancer",
        "accidents",
        # "stroke",  # Not separate in WHO
        "respiratory",
        # "alzheimers",  # Not separate in WHO
        "diabetes",
        "kidney",
        "liver",
        # "covid",  # Not separate in WHO 2020 data (included in respiratory infections)
        "suicide",
        "influenza",
        # "drug overdose",  # Not in WHO
        # "homicide",  # Not separate in WHO (part of intentional injuries)
        "terrorism",
    ],

    # WHO data doesn't need mapping (handled by load_who_data.py)
    "causes_map": {},

    # Colors for visualization
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
    },

    # API settings
    "api_sleep": 10,

    # ========================================================================
    # MEDIA CLOUD QUERY KEYWORDS (Portuguese/Brazilian)
    # ========================================================================
    # These keywords are translated to Brazilian Portuguese.
    # Medical terminology may vary between Brazil and Portugal.
    # ========================================================================
    "queries": {
        "heart disease": {
            "single_terms": [
                "doença cardíaca",
                "doença do coração",
                "ataque cardíaco",
                "ataque do coração",
                "infarto",
                "parada cardíaca",
                "arritmia",
                "insuficiência cardíaca",
                "doença arterial coronariana",
                "hipertensão",
                "pressão alta",
                "cardiologia",
                "cardiologista",
            ],
            "combinations": [
                "doença cardíaca coração",
                "ataque cardíaco coração",
                "coração infarto",
                "coração cardíaco",
                "cardíaco cardíaco",
                "coração arritmia",
                "arritmia cardíaco",
                "insuficiência cardíaca coração",
                "hipertensão cardíaco",
                "pressão alta coração",
                "cardiologia coração",
            ],
            "exclude_terms": [],
        },

        "cancer": {
            "single_terms": [
                "câncer",
                "cancro",  # Less common in Brazil, more in Portugal
                "tumor",
                "carcinoma",
                "sarcoma",
                "leucemia",
                "linfoma",
                "melanoma",
                "oncologia",
                "oncologista",
                "quimioterapia",
                "radioterapia",
                "imunoterapia",
                "biópsia",
                "metástase",
                "remissão",
                "carcinogênico",
            ],
            "combinations": [
                "câncer câncer",
                "câncer tumor",
                "câncer carcinoma",
                "câncer oncologia",
                "câncer quimioterapia",
                "câncer metástase",
                "tumor tumor",
            ],
            "exclude_terms": [],
        },

        "accidents": {
            "single_terms": [
                "acidente de carro",
                "acidente de trânsito",
                "acidente automobilístico",
                "colisão",
                "batida de carro",
                "atropelamento",
                "acidente de moto",
                "acidente de avião",
                "acidente de trem",
                "acidente de trabalho",
                "acidente industrial",
                "eletrocussão",
                "queimadura",
                "afogamento",
                "incêndio",
                "fogo",
            ],
            "combinations": [
                "acidente acidente",
                "acidente carro",
                "acidente trânsito",
                "colisão carro",
                "fogo morreu",
                "fogo morte",
                "incêndio morte",
            ],
            "exclude_terms": [],
        },

        "stroke": {
            "single_terms": [
                "derrame cerebral",
                "AVC",
                "acidente vascular cerebral",
                "derrame",
                "isquemia cerebral",
                "hemorragia cerebral",
                "embolia cerebral",
                "neurologia",
            ],
            "combinations": [
                "derrame cerebral",
                "AVC cerebral",
                "acidente vascular cerebral",
                "derrame cérebro",
                "isquemia cerebral",
                "hemorragia cerebral",
            ],
            "exclude_terms": [],
        },

        "respiratory": {
            "single_terms": [
                "doença pulmonar obstrutiva crônica",
                "DPOC",
                "bronquite crônica",
                "enfisema",
                "asma",
                "insuficiência respiratória",
                "doença pulmonar",
                "doença respiratória",
                "infecção respiratória",
            ],
            "combinations": [
                "DPOC pulmão",
                "DPOC respiratória",
                "enfisema pulmão",
                "asma pulmão",
                "respiratória pulmão",
            ],
            "exclude_terms": [],
        },

        "alzheimers": {
            "single_terms": [
                "Alzheimer",
                "Alzheimer's",
                "doença de Alzheimer",
                "demência",
            ],
            "combinations": [
                "Alzheimer Alzheimer",
                "Alzheimer demência",
                "demência demência",
            ],
            "exclude_terms": [],
        },

        "diabetes": {
            "single_terms": [
                "diabetes",
                "diabético",
                "insulina",
                "hiperglicemia",
                "glicemia alta",
            ],
            "combinations": [
                "diabetes diabetes",
                "diabetes insulina",
                "diabetes diabético",
                "insulina diabético",
            ],
            "exclude_terms": [],
        },

        "kidney": {
            "single_terms": [
                "doença renal",
                "insuficiência renal",
                "falência renal",
                "diálise",
                "hemodiálise",
                "nefropatia",
                "nefrologia",
                "nefrologista",
                "rim",
            ],
            "combinations": [
                "doença renal rim",
                "insuficiência renal rim",
                "diálise rim",
                "hemodiálise rim",
                "nefrologia rim",
            ],
            "exclude_terms": [],
        },

        "liver": {
            "single_terms": [
                "doença hepática",
                "cirrose",
                "hepatite",
                "insuficiência hepática",
                "fígado gorduroso",
                "hepatologia",
                "hepatologista",
                "transplante de fígado",
                "fígado",
            ],
            "combinations": [
                "doença hepática fígado",
                "cirrose fígado",
                "hepatite hepatite",
                "hepatite fígado",
                "insuficiência hepática fígado",
                "transplante fígado",
            ],
            "exclude_terms": [],
        },

        "covid": {
            "single_terms": [
                "COVID-19",
                "COVID",
                "coronavírus",
                "coronavirus",
                "SARS-CoV-2",
                "covid",
            ],
            "combinations": [
                "COVID-19 COVID-19",
                "coronavírus coronavírus",
                "COVID coronavírus",
                "SARS-CoV-2 COVID",
            ],
            "exclude_terms": [],
        },

        "suicide": {
            "single_terms": [
                "suicídio",
                "suicida",
                "autolesão",
                "autoflagelo",
                "lesão autoinfligida",
            ],
            "combinations": [
                "suicídio suicídio",
                "suicídio suicida",
                "suicídio morte",
                "suicídio depressão",
                "depressão suicida",
            ],
            "exclude_terms": [],
        },

        "influenza": {
            "single_terms": [
                "influenza",
                "gripe",
                "H1N1",
                "pneumonia",
                "infecção respiratória",
                "infecção pulmonar",
            ],
            "combinations": [
                "influenza gripe",
                "gripe pulmão",
                "gripe respiratória",
                "pneumonia pulmão",
                "pneumonia respiratória",
            ],
            "exclude_terms": [],
        },

        "drug overdose": {
            "single_terms": [
                "overdose",
                "sobredose",
                "uso de drogas",
                "dependência química",
                "vício em drogas",
                "abuso de substâncias",
                "crack",
                "cocaína",
                "heroína",
                "opioides",
                "drogas",
            ],
            "combinations": [
                "overdose drogas",
                "sobredose drogas",
                "overdose morte",
                "dependência química drogas",
                "vício drogas",
                "abuso substâncias drogas",
                "cocaína overdose",
            ],
            "exclude_terms": [],
        },

        "homicide": {
            "single_terms": [
                "homicídio",
                "assassinato",
                "assassino",
                "morte violenta",
                "crime violento",
                "violência",
                "tiroteio",
                "bala perdida",
                "fuzilamento",
                "execução",
                "esfaqueamento",
                "facada",
                "latrocínio",
            ],
            "combinations": [
                "homicídio homicídio",
                "homicídio assassinato",
                "homicídio morte",
                "assassinato morte",
                "crime violento morte",
                "violência morte",
                "tiroteio morte",
                "tiroteio violência",
            ],
            "exclude_terms": [],
        },

        "terrorism": {
            "single_terms": [
                "terrorismo",
                "terrorista",
                "ataque terrorista",
                "atentado",
                "extremismo",
                "bomba",
                "explosão terrorista",
            ],
            "combinations": [
                "terrorismo terrorista",
                "terrorista ataque",
                "terrorismo atentado",
                "terrorista morte",
                "atentado terrorista",
                "extremismo terrorismo",
            ],
            "exclude_terms": [],
        },
    },
}
