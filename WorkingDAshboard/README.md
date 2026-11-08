# NIRF Insights Dashboard

An interactive Streamlit dashboard for exploring India's National Institutional
Ranking Framework (NIRF) data from **2017 to 2025**.

The dashboard supports exploratory charts, the 40 prescribed analysis
questions from the assignment, and an optional Gemini-powered natural-language
assistant.

## Features

- **Overview**: headline statistics and highlights for the latest ranking year.
- **Graph Builder**: create custom visualisations by selecting ranking
  categories, years, institutions, states, and NIRF parameters.
- **Annexure A**: browse all 40 questions across eight analytical themes, with
  charts and data-driven findings.
- **Ask AI**: ask questions about the NIRF dataset in plain English. The
  assistant queries the dashboard data and can generate a chart when useful.
- CSV download support for tables generated in the Graph Builder.
- Cached data loading for faster page navigation.

## Technology

- Python
- Streamlit
- Pandas and NumPy
- Plotly
- Matplotlib and Seaborn
- SciPy and Statsmodels
- Google GenAI SDK (optional AI assistant)

## Requirements

- Python 3.10 or newer
- Internet access for installing dependencies
- A Gemini API key only if the **Ask AI** page is required

## Installation

Open a terminal in this directory:

```bash
cd "C:\Users\sajja\OneDrive\Desktop\DHV Assignment\WorkingDAshboard\WorkingDAshboard"
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the dashboard

Start Streamlit with:

```bash
python -m streamlit run app.py
```

Then open the URL shown in the terminal, usually
`http://localhost:8501`.

On Windows, `start_dashboard.bat` can be used as a convenience launcher. It
checks for Python and required packages, chooses an available port, and opens
the dashboard in a browser.

## Enabling Ask AI

1. Copy `.env.example` to `.env`.
2. Add a Gemini API key:

   ```env
   GEMINI_API_KEY=your-key-here
   ```

3. Optionally configure the primary and fallback models using
   `GEMINI_MODEL` and `GEMINI_FALLBACK_MODELS`.
4. Restart Streamlit.

The rest of the dashboard works without a Gemini API key. Do not commit `.env`
or share the API key.

## Data

The dashboard reads `nirf_cleaned_2017_2025.csv` from the project directory.
The main fields include:

| Field | Description |
|---|---|
| `Year` | Ranking year |
| `Category` | NIRF category, such as Overall or Engineering |
| `Institute ID` | NIRF institution identifier |
| `Name` | Institution name |
| `City`, `State` | Institution location |
| `Score` | Overall NIRF score |
| `Rank`, `Rank_Num`, `Rank_Band` | Published and numeric rank representations |
| `TLR` | Teaching, Learning and Resources score |
| `RPC` | Research and Professional Practice score |
| `GO` | Graduation Outcomes score |
| `OI` | Outreach and Inclusivity score |
| `Perception` | Perception score |

State names are normalised when the data is loaded so that inconsistent
capitalisation does not split the same state into multiple filter values.

## Project structure

```text
app.py                    Streamlit entry point and page navigation
data.py                   Shared cached data-loading entry point
questions_visualizer.py   Annexure A question catalogue and visualisations
graph_builder.py          Graph Builder controls and table generation
dynamic_charts.py         Plotly chart implementations
chatbot_engine.py         Gemini assistant and dataset query tools
chat_ui.py                Assistant interface
ui.py                     Shared layout, styling, and UI helpers
views/
  overview.py             Overview page
  builder.py              Graph Builder page
  annexure.py             Annexure A page
  assistant.py            Ask AI page
assets/logo.svg           Dashboard logo
```

## Notes

- The dashboard is designed for local use and presentation.
- Charts use the bundled CSV file; replacing the CSV may require matching the
  expected column names.
- The AI assistant is restricted to questions about the NIRF dataset and
  higher-education rankings.
