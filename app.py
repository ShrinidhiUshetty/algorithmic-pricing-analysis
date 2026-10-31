

import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Algorithmic Pricing Analysis",
    page_icon="📊",
    layout="wide"
)

# Dashboard title
st.title("📊 Algorithmic Pricing Analysis")
st.subheader("Causal Inference & Synthetic Control Engine")

st.info(
    "This project explores Amazon sales data and uses "
    "a separate simulated experiment to estimate "
    "the impact of price changes on sales."
)

# Sidebar navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Analysis",
    [
        "Project Overview",
        "Exploratory Data Analysis",
        "Causal Inference",
        "Model Validation",
        "Pricing Insights"
    ]
)

# Load simulated dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/regional_sales_simulated.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# Page 1: Project Overview
if page == "Project Overview":

    st.header("Project Overview")

    st.write(
        "This project investigates how pricing decisions "
        "affect sales using data analytics and causal "
        "inference techniques."
    )

    st.markdown("### Project Objectives")

    st.write(
        "1. Explore Amazon sales and pricing patterns."
    )
    st.write(
        "2. Analyze the relationship between price and sales."
    )
    st.write(
        "3. Implement Difference-in-Differences (DiD)."
    )
    st.write(
        "4. Develop a Synthetic Control model."
    )
    st.write(
        "5. Evaluate pricing interventions and model robustness."
    )

    st.markdown("### Technologies Used")

    st.write(
        "Python, Pandas, NumPy, Matplotlib, SciPy, "
        "Scikit-learn, Google Colab, GitHub and Streamlit."
    )


# Page 2: Exploratory Data Analysis
elif page == "Exploratory Data Analysis":

    st.header("Exploratory Data Analysis")

    st.write(
        "The EDA stage explores an Amazon sales dataset "
        "containing 100,000 records and 20 columns."
    )

    st.write(
        "The analysis includes data cleaning, monthly "
        "sales trends, product pricing, discounts, "
        "and price-quantity correlations."
    )

    st.info(
        "Amazon EDA charts will be added in the next stage."
    )


# Page 3: Causal Inference
elif page == "Causal Inference":

    st.header("Causal Inference Results")

    df = load_data()

    st.markdown("### Simulated Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", len(df))

    with col2:
        st.metric("Regions", df["Region"].nunique())

    with col3:
        st.metric("Days", df["Date"].nunique())

    st.markdown("### Regional Sales Trends")

    selected_regions = st.multiselect(
        "Select regions to compare",
        options=sorted(df["Region"].unique()),
        default=["Bengaluru", "Hyderabad"]
    )

    if selected_regions:

        filtered_df = df[
            df["Region"].isin(selected_regions)
        ]

        chart_data = filtered_df.pivot(
            index="Date",
            columns="Region",
            values="Sales"
        )

        st.line_chart(chart_data)

    else:
        st.warning("Please select at least one region.")

    st.caption(
        "Source: Simulated regional sales dataset. "
        "Bengaluru received a 10% price reduction "
        "starting January 31, 2024."
    )

    st.markdown("### Regional Sales Data")

    st.dataframe(df, use_container_width=True)


# Page 4: Model Validation
elif page == "Model Validation":

    st.header("Model Validation")

    st.write(
        "The causal models were evaluated using "
        "placebo tests, pre-intervention model fit, "
        "and sensitivity analysis."
    )

    st.info(
        "Detailed validation charts will be added "
        "in the next stage."
    )


# Page 5: Pricing Insights
elif page == "Pricing Insights":

    st.header("Pricing and Business Insights")

    st.write(
        "This section will explain the estimated "
        "impact of pricing interventions on sales."
    )

    st.info(
        "Pricing insights and elasticity visualizations "
        "will be added in the next stage."
    )


# Footer
st.divider()

st.caption(
    "Algorithmic Pricing Analysis | "
    "Data Analytics Project | "
    "Causal results use simulated data."
)

