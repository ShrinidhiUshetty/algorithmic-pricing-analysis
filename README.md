# Algorithmic Pricing Analysis
### Causal Inference & Synthetic Control Engine for Algorithmic Pricing

An end-to-end data analytics project that evaluates the impact of a simulated price reduction on regional sales using **Python, Exploratory Data Analysis (EDA), Difference-in-Differences (DiD), Synthetic Control, and Streamlit**.

The project estimates how a 10% price reduction in Bengaluru affected daily sales compared with a model-generated counterfactual.

## Project Objectives

- Analyze regional sales and pricing patterns.
- Estimate the impact of a pricing intervention.
- Compare actual sales with expected sales without the intervention.
- Validate the Synthetic Control model using diagnostic checks.
- Evaluate price elasticity and estimated sales uplift.
- Present findings through an interactive Streamlit dashboard.

## Technologies Used

- **Python:** Data processing and analysis
- **Pandas & NumPy:** Data manipulation and calculations
- **Matplotlib & Plotly:** Data visualization
- **Streamlit:** Interactive dashboard
- **Google Colab:** Data analysis and model development
- **GitHub:** Version control and documentation

## Dataset Description

The project uses simulated regional sales data.

| Dataset | Description |
|---|---|
| `regional_sales_simulated.csv` | 240 observations across four regions |
| `synthetic_control_results.csv` | 60 daily actual and synthetic Bengaluru observations |

**Regions:** Bengaluru, Hyderabad, Chennai, Pune

**Analysis Period:** January 1 – February 29, 2024

**Intervention Date:** January 31, 2024

**Pricing Intervention:** Bengaluru price reduced from 100 to 90 (10% reduction).

## Project Methodology

### 1. Exploratory Data Analysis

Analyzed regional sales trends, average prices, monthly sales performance, and the relationship between price and demand.

### 2. Difference-in-Differences

Compared Bengaluru's sales changes before and after the intervention with changes in the control regions.

### 3. Synthetic Control Method

Used a weighted combination of control regions to estimate Bengaluru's sales without the price intervention.

### 4. Model Validation

Evaluated model fit and robustness using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Pre-intervention bias
- Temporal placebo checks
- Training-window sensitivity analysis
- Comparison between DiD and Synthetic Control estimates

### 5. Pricing Impact Analysis

Estimated additional daily sales, percentage sales uplift, and approximate price elasticity.

## Key Project Results

| Metric | Result |
|---|---:|
| Price Reduction | 10% |
| Estimated Additional Daily Sales | +20.33 units |
| Estimated Total Additional Sales | +610.01 units |
| Estimated Sales Uplift | +9.97% |
| Pre-Intervention RMSE | 4.81 |
| Approximate Adjusted Price Elasticity | −1.00 |

These results are based on the original simulated experiment and saved model outputs. They are not dynamically refitted when dashboard filters change.

## Interactive Streamlit Dashboard

The dashboard contains six sections:

**1. Executive Overview**
- Sales performance KPIs
- Regional sales trends
- Synthetic Control headline results

**2. Exploratory Analysis**
- Daily sales trends
- Regional pricing comparisons
- Price versus sales analysis
- Monthly sales summary

**3. Causal Impact**
- Actual versus Synthetic Bengaluru
- Daily treatment effects
- Estimated sales uplift
- Difference-in-Differences comparison

**4. Model Diagnostics**
- RMSE, MAE, and bias
- Actual versus synthetic sales
- DiD versus Synthetic Control
- Temporal placebo checks
- Training-window sensitivity

**5. Pricing Insights**
- Regional pricing scorecard
- Revenue proxy comparison
- Expected versus observed sales
- Estimated sales uplift
- Approximate adjusted price elasticity
- Business interpretation

**6. Data Explorer**
- Regional sales dataset
- Synthetic Control dataset
- Interactive filtering
- CSV download options

## Project Structure

```text
algorithmic-pricing-analysis/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── regional_sales_simulated.csv
│   └── synthetic_control_results.csv
└── notebooks/
    ├── 01_data_cleaning.ipynb
    └── 02_causal_inference.ipynb
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Business Insights

The simulated price reduction was associated with an estimated increase in Bengaluru's daily sales volume.

The Synthetic Control analysis estimated approximately 20 additional units per day, corresponding to a 9.97% uplift relative to the counterfactual.

The approximate adjusted elasticity of −1.00 suggests a roughly proportional percentage change in estimated demand relative to the percentage price change.

However, increased sales volume does not necessarily mean increased revenue or profit. Pricing decisions should also consider margins, costs, and other business factors.

## Limitations

- All sales data is simulated.
- Causal estimates depend on modeling assumptions.
- The Streamlit dashboard uses saved Synthetic Control results rather than retraining the model.
- Placebo and sensitivity results are based on the original analysis.
- The revenue proxy is not actual transaction-level revenue.
- Results should not be interpreted as verified real-world business outcomes.

## Conclusion

This project demonstrates how data analytics and causal inference techniques can support pricing decisions.

By combining EDA, Difference-in-Differences, Synthetic Control, model diagnostics, and interactive visualization, the project provides a structured approach to evaluating the potential impact of pricing interventions.

**Project Status: Completed — Ready for final verification and submission.**
