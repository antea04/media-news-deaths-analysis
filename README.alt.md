# Media Deaths Analysis

Analyze media coverage of different causes of death by comparing mentions in news outlets against actual mortality statistics.

**Key Feature**: Multi-language/region support with a plugin architecture - add new countries in 4 simple steps.

For methodology details, see the [methodology document](https://docs.owid.io/projects/etl/analyses/media_deaths/methodology/).

---

## Quick Start

```bash
# Install dependencies
uv sync

# Set up your Media Cloud API token
cp .env.example .env
# Edit .env and add your token

# List available configurations
uv run media-deaths --list-languages

# Run analysis
uv run media-deaths ca_ES
```

---

## Installation

### Prerequisites

- **Python 3.11+**
- **[uv](https://github.com/astral-sh/uv)** package manager
  ```bash
  # Test if installed
  uv --version
  ```

### Get Media Cloud API Key

1. Create a free account at [Media Cloud](https://search.mediacloud.org/sign-up)
2. Follow the [API setup tutorial](https://github.com/mediacloud/api-tutorial-notebooks/blob/main/MC01%20-%20setup.ipynb) to generate your API token
3. Copy your API token for the next step

### Setup

```bash
# Clone and install
git clone <repository-url>
cd media-news-deaths-analysis
uv sync

# Configure API token
cp .env.example .env
# Edit .env and add your token

# Verify
uv run media-deaths --list-languages
```

---

## How It Works

This tool queries news outlets for mentions of different causes of death (e.g., cancer, heart disease, homicide) and compares the media coverage against actual mortality statistics from official health data sources.

**Key insight**: Media coverage often doesn't match reality - some causes are over-reported while others are under-reported relative to their actual frequency.

**Multi-language support**: The project is organized as a plugin system where each language/region is self-contained. Currently supports **Catalan (Spain)** with the `ca_ES` configuration ([see example files](configs/cat.yml)). Adding new languages requires no changes to core code - just create a config file and data loader following the existing structure.

---

## Usage

```bash
# List available configurations
uv run media-deaths --list-languages

# Run analysis
uv run media-deaths ca_ES

# Test with specific causes
uv run media-deaths ca_ES --causes cancer "heart disease"

# Custom config file
uv run media-deaths --custom-config path/to/config.yml

# Custom output directory
uv run media-deaths ca_ES --output-dir ./results
```

**Output files** (saved to `./data/`):
- `media_deaths_mentions_{language}.csv` - Raw API results
- `media_deaths_results_{language}.csv` - Analysis results
- `media_deaths_plot_{language}.png` - Visualization

**Performance**: Time varies by config (~0.5 min per query). Example: 3 causes × 4 sources = 12 queries = ~6 min. Cached runs: <1 min (set `rerun_queries: false`)

---

## HOWTO add a new language/region

Taking the `ca_ES` example (Catalan in Spain), you need to follow the following **4 steps**:

### Step 1: Create Config File

Create `configs/your_country.yml` following the structure in [`configs/cat.yml`](configs/cat.yml):

**Key sections**:
- `runtime`: Analysis settings (see [`configs/cat.yml:4-18`](configs/cat.yml#L4-L18))
- `data`: Year, language, data loader name
- `outlets`: Media outlets to query (find IDs at [search.mediacloud.org](https://search.mediacloud.org))
- `collections`: Geographic collections
- `causes_death`: Translated search terms for each cause

See full example structure in [`configs/cat.yml`](configs/cat.yml).

### Step 2: Create Data Loader

Create `configs/data_loaders/your_loader.py` following [`configs/data_loaders/catalan_gencat.py`](configs/data_loaders/catalan_gencat.py):

**Key requirements**:
- Use `@register_loader("unique_name")` decorator
- Return DataFrame with columns: `["cause", "deaths", "year"]`
- Map local cause names to causes in your config file
- Include only one year's data

**Structure** (see [`catalan_gencat.py:14-43`](configs/data_loaders/catalan_gencat.py#L14-L43)):
1. `load_data()` - Main function with decorator
2. `_fetch_data()` - Get data from source (CSV, Excel, API)
3. `_clean_data()` - Format and clean
4. `_aggregate_causes()` - Map to standard cause categories

### Step 3: Register in Index

Add entry to [`configs/index.yml`](configs/index.yml):

```yaml
your_CODE:
  name: Your Language (Country)
  config: your_country.yml
  description: Brief description
```

Format: `ll_CC` (ISO-639 language + ISO-3166 country), e.g., `fr_FR`, `en_US`, `es_MX`

### Step 4: Import Loader

Add import to [`configs/data_loaders/__init__.py`](configs/data_loaders/__init__.py):

```python
from . import your_loader  # noqa: F401
```

### Test

```bash
# Verify it appears
uv run media-deaths --list-languages

# Test with one cause
uv run media-deaths your_CODE --causes cancer

# Full run
uv run media-deaths your_CODE
```

---

## Tips

**Finding Media Outlets**: Browse [Geographic Collections](https://search.mediacloud.org/collections/news/geographic) or search at [search.mediacloud.org](https://search.mediacloud.org)

**Translating Queries**: Include medical terms AND colloquial terms. Test at [search.mediacloud.org](https://search.mediacloud.org)

**Config Reference**: See full examples in [`configs/cat.yml`](configs/cat.yml)

**Performance**: Set `rerun_queries: false` to use cached data. Time estimate shown at start (formula: ~0.5 min × causes × sources)

---

## Troubleshooting

**API Rate Limiting**: Increase `api_sleep` in config (e.g., from 10 to 15 seconds)

**Data Loader Issues**: Test directly:
```bash
python3 -c "from configs.data_loaders.your_loader import load_data; print(load_data())"
```

**Config Validation**: Test loading:
```bash
python3 -c "from media_deaths.config import Config; c = Config('configs/your.yml'); print(c.YEAR)"
```

---

## Methodology

Compares media mentions vs actual mortality statistics. See [methodology document](https://docs.owid.io/projects/etl/analyses/media_deaths/methodology/) for details.
