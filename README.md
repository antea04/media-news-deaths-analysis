# Media Deaths Analysis

This project analyzes media coverage of different causes of death by querying the Media Cloud API and comparing mentions in major news outlets against actual death statistics.

For more details, see the [methodology document](https://docs.owid.io/projects/etl/analyses/media_deaths/methodology/).

## Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager (test if installed: `uv --version`)

### Getting a Media Cloud API Key

To run queries against the Media Cloud database, you'll need an API token:

1. Create a free account at [Media Cloud](https://search.mediacloud.org/sign-up)
2. Follow the instructions in the [API setup tutorial](https://github.com/mediacloud/api-tutorial-notebooks/blob/main/MC01%20-%20setup.ipynb) to generate your API token
3. Copy your API token for the next step

### Environment Setup

1. **Clone or download this repository**

2. **Create environment file:**
   ```bash
   cp .env.example .env
   ```

3. **Add your Media Cloud API token:**

   Open the `.env` file and replace `your_api_token_here` with your actual API token:
   ```
   MC_API_TOKEN=your_actual_token_here
   ```

4. **Install dependencies:**
   ```bash
   uv sync
   ```

5. **Run the analysis:**
   ```bash
   uv run python media_deaths_analysis.py
   ```

## Project Structure

- **[media_deaths_analysis.py](media_deaths_analysis.py)** - Main script that loads death statistics, queries Media Cloud API, analyzes data, and generates visualizations
- **[query_generation.py](query_generation.py)** - Defines search keywords and creates Boolean queries for each cause of death

## Configuration

The main script includes several configuration options at the top of [media_deaths_analysis.py](media_deaths_analysis.py):

- `YEAR` - Year to analyze (default: 2023)
- `LANGUAGE` - Language code for analysis (default: "en")
- `RERUN_QUERIES` - Whether to query Media Cloud API or use cached data (default: True)
- `RUN_SINGLE_QUERIES` - Whether to run single-keyword queries in addition to full queries (default: False)
- `USE_SAVED_RESULTS` - Whether to load pre-computed results (default: False)
- `OVERWRITE` - Whether to overwrite existing result files (default: True)
- `CAUSES_OF_DEATH` - List of causes of death to analyze
- `OUTLETS` - List of media outlets to query (NYT, Washington Post, Fox News)
- `COLLECTIONS` - List of Media Cloud collections to query (US Collection)

## Localization Tasks

This analysis can be adapted to different countries and languages. Here are the steps to localize the analysis:

### 1. Find Death Statistics Data for Your Country

**Files to modify:** [media_deaths_analysis.py](media_deaths_analysis.py)

**Methods that need changes:**
- `format_death_data()` at [media_deaths_analysis.py:127-183](media_deaths_analysis.py#L127-L183) - Replace CDC data sources with your country's death statistics
- `create_tb_death()` at [media_deaths_analysis.py:186-231](media_deaths_analysis.py#L186-L231) - Adjust data processing logic for your country's data format

**What to change:**
- Replace the CDC Wonder database URLs (lines 650-658) with your country's death statistics source
- Update the data loading format to match your source (CSV, Excel, API, etc.)

### 2. Map Death Statistics to Categories

**Files to modify:** [media_deaths_analysis.py](media_deaths_analysis.py)

**Methods that need changes:**
- `format_death_data()` at [media_deaths_analysis.py:127-183](media_deaths_analysis.py#L127-L183) - Update the `CAUSES_MAP` dictionary (lines 137-150)

**What to change:**
- Update `CAUSES_MAP` to map your country's cause-of-death categories to the standardized keywords
- You can refer to the [methodology document](https://docs.owid.io/projects/etl/analyses/media_deaths/methodology/) for CDC category mappings
- Optionally, modify `CAUSES_OF_DEATH` list (lines 64-80) to include or exclude specific causes relevant to your country

### 3. Translate Query Terms into Target Language

**Files to modify:** [query_generation.py](query_generation.py)

**Methods that need changes:**
- `create_queries()` at [query_generation.py:80-838](query_generation.py#L80-L838) - Translate all search terms for each cause of death

**What to change:**
- Translate `single_terms` lists for each cause (e.g., "heart disease" → "maladie cardiaque" for French)
- Translate `combinations` lists (paired keywords used in proximity searches)
- Update `exclude_terms` lists with language-appropriate exclusions
- Consider cultural/linguistic differences (e.g., different medical terminology, colloquial terms)

**Methods that should NOT be changed:**
- `create_query_str()` at [query_generation.py:48-70](query_generation.py#L48-L70) - Query building logic remains the same
- `create_queries_by_cause()` at [query_generation.py:73-77](query_generation.py#L73-L77) - Structure is language-agnostic
- `create_full_queries()` at [query_generation.py:30-33](query_generation.py#L30-L33) - Wrapper function stays the same
- `create_single_keyword_queries()` at [query_generation.py:36-45](query_generation.py#L36-L45) - Logic is universal

### 4. Choose Media Outlets or Collections

**Files to modify:** [media_deaths_analysis.py](media_deaths_analysis.py)

**Configuration to change:**
- `OUTLETS` list at [media_deaths_analysis.py:113-117](media_deaths_analysis.py#L113-L117) - Add your country's newspapers
- `COLLECTIONS` list at [media_deaths_analysis.py:119](media_deaths_analysis.py#L119) - Add geographic collections for your country
- Update `LANGUAGE` variable at [media_deaths_analysis.py:34](media_deaths_analysis.py#L34) (e.g., "fr", "es", "de")

**How to find outlets:**
- Browse [Media Cloud Geographic Collections](https://search.mediacloud.org/collections/news/geographic)
- Search for specific newspapers (e.g., Le Monde, El País, Die Zeit)
- Use Media Cloud's search interface to find source IDs for your outlets

**Example configuration:**
```python
OUTLETS = [
    {"full_name": "Le Monde", "id": 12345, "short_name": "lemonde"},
    {"full_name": "El País", "id": 67890, "short_name": "elpais"},
]
COLLECTIONS = [
    {"full_name": "France Collection", "id": 98765, "short_name": "france"}
]
```

### 5. Query the Database

**Methods that should NOT be changed:**
- `query_results()` at [media_deaths_analysis.py:239-265](media_deaths_analysis.py#L239-L265) - API querying logic
- `get_mentions_from_source()` at [media_deaths_analysis.py:268-321](media_deaths_analysis.py#L268-L321) - Query execution
- `get_media_mentions()` at [media_deaths_analysis.py:324-430](media_deaths_analysis.py#L324-L430) - Overall query orchestration

**Tips:**
- Start with a subset of causes (modify `CAUSES_OF_DEATH` list) to test before running all queries
- Test with just 1 outlet first to avoid long API wait times
- Adjust `API_SLEEP` at [media_deaths_analysis.py:39](media_deaths_analysis.py#L39) if you hit rate limits (increase the wait time)

### 6. Analyze and Plot Results

**Methods that should NOT be changed:**
- `add_shares()` at [media_deaths_analysis.py:438-459](media_deaths_analysis.py#L438-L459) - Share calculation logic
- `analyze_data()` at [media_deaths_analysis.py:462-535](media_deaths_analysis.py#L462-L535) - Data analysis
- `plot_media_deaths_matplotlib()` at [media_deaths_analysis.py:543-619](media_deaths_analysis.py#L543-L619) - Plotting function

**Optional customization:**
- Update `FIXED_COLOURS` dictionary at [media_deaths_analysis.py:83-104](media_deaths_analysis.py#L83-L104) if you want different colors
- Modify visualization titles and labels to your language
- Consider alternative visualization tools (e.g., [RAWGraphs](https://www.rawgraphs.io/)) using the CSV output

### 7. Share Your Results!

After completing your localized analysis:
- Share the visualizations and findings e.g. in the google doc
- Consider opening a PR with localized queries 
- Document any insights specific to your country's media coverage patterns in google Doc

### Quick Start for Testing

To test localization with minimal API calls:

1. Keep only one cause in `CAUSES_OF_DEATH` (e.g., "heart disease")
2. Set one outlet in `OUTLETS`
3. Set `RERUN_QUERIES=True` and `RUN_SINGLE_QUERIES=False`
4. Run the script to verify translations
5. Once validated, expand to all causes and outlets

## Notes

- Running queries takes ~30 minutes due to API rate limits (10 seconds between requests)
- Set `RERUN_QUERIES=False` to use cached data
- Adjust configuration options at the top of [media_deaths_analysis.py](media_deaths_analysis.py) (year, language, outlets, etc.)
