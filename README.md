# Algorithmic Pricing Analysis — Streamlit Dashboard

Interactive dashboard built around the two CSV datasets supplied in the original ZIP.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Dashboard sections
- **Executive Overview:** KPI cards, regional trends, sales distribution, synthetic-control headline estimate.
- **Exploratory Analysis:** time series, regional price comparison, price-sales scatter, monthly summary.
- **Causal Impact:** actual vs synthetic Bengaluru, daily effect, configurable intervention date, DiD cross-check.
- **Model Diagnostics:** pre/post MAE and RMSE, residual chart, interpretation caveats.
- **Pricing Insights:** regional scorecard, revenue proxy, Bengaluru before/after comparison.
- **Data Explorer:** filtered data tables and CSV downloads.

## Data and assumptions
- `data/regional_sales_simulated.csv`: 240 rows, four regions, daily observations from 2024-01-01 through 2024-02-29.
- `data/synthetic_control_results.csv`: 60 daily Bengaluru actual and synthetic observations.
- The project data is simulated. The intervention date defaults to 2024-01-31, consistent with the original app's stated assumption.
- The app computes estimates from the supplied data; it does not refit the synthetic-control model.
- The difference-in-differences estimate assumes a treated region (Bengaluru) and uses the other observed regions as controls. Parallel trends, no spillovers, and other identification assumptions are not verified.
- Price-sales correlation and before/after comparisons are descriptive, not causal.
- Revenue proxy is average price multiplied by total units; transaction-level revenue is not present in the supplied data.

## Project structure

```text
algorithmic-pricing-analysis-streamlit/
├── app.py
├── requirements.txt
├── README.md
└── data/
    ├── regional_sales_simulated.csv
    └── synthetic_control_results.csv
```
