
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Algorithmic Pricing Analysis",
    page_icon="📊",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

DEFAULT_INTERVENTION = pd.Timestamp("2024-01-31")


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    regional = pd.read_csv(
        DATA_DIR / "regional_sales_simulated.csv"
    )

    synthetic = pd.read_csv(
        DATA_DIR / "synthetic_control_results.csv"
    )

    regional["Date"] = pd.to_datetime(
        regional["Date"], errors="coerce"
    )

    synthetic["Date"] = pd.to_datetime(
        synthetic["Date"], errors="coerce"
    )

    regional = regional.dropna(
        subset=["Date", "Region", "Price", "Sales"]
    ).copy()

    synthetic = synthetic.dropna(
        subset=[
            "Date",
            "Bengaluru",
            "Synthetic_Bengaluru"
        ]
    ).copy()

    return regional, synthetic


try:
    df, sc = load_data()

except Exception as exc:
    st.error(f"Could not load project datasets: {exc}")
    st.stop()


required_regional = {
    "Date", "Region", "Price", "Sales"
}

required_synthetic = {
    "Date", "Bengaluru", "Synthetic_Bengaluru"
}

if (
    not required_regional.issubset(df.columns)
    or not required_synthetic.issubset(sc.columns)
):
    st.error("Dataset columns do not match the expected format.")
    st.stop()

if df.empty or sc.empty:
    st.error("The project datasets contain no usable observations.")
    st.stop()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 Pricing Analytics")

page = st.sidebar.radio(
    "Workspace",
    [
        "Executive Overview",
        "Exploratory Analysis",
        "Causal Impact",
        "Model Diagnostics",
        "Pricing Insights",
        "Data Explorer"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("Global Filters")

regions = sorted(df["Region"].unique().tolist())

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

date_range = st.sidebar.date_input(
    "Analysis period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if (
    isinstance(date_range, (tuple, list))
    and len(date_range) == 2
):
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])

else:
    start_date = pd.Timestamp(min_date)
    end_date = pd.Timestamp(max_date)

selected_regions = st.sidebar.multiselect(
    "Regions",
    options=regions,
    default=regions
)

intervention_date = st.sidebar.date_input(
    "Price intervention date",
    value=DEFAULT_INTERVENTION.date(),
    min_value=min_date,
    max_value=max_date
)

intervention_date = pd.Timestamp(intervention_date)

st.sidebar.caption(
    "Original experiment: Bengaluru price reduced "
    "from 100 to 90 on January 31, 2024."
)

st.sidebar.caption(
    "All project datasets are simulated."
)


# ============================================================
# FILTER DATA
# ============================================================

filtered = df[
    (df["Date"] >= start_date)
    & (df["Date"] <= end_date)
    & (df["Region"].isin(selected_regions))
].copy()

sc_view = sc[
    (sc["Date"] >= start_date)
    & (sc["Date"] <= end_date)
].copy()

sc_view["Effect"] = (
    sc_view["Bengaluru"]
    - sc_view["Synthetic_Bengaluru"]
)

pre_sc = sc_view[
    sc_view["Date"] < intervention_date
].copy()

post_sc = sc_view[
    sc_view["Date"] >= intervention_date
].copy()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_errors(data):
    if data.empty:
        return {
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias": np.nan
        }

    errors = (
        data["Bengaluru"]
        - data["Synthetic_Bengaluru"]
    )

    return {
        "RMSE": np.sqrt(np.mean(errors ** 2)),
        "MAE": np.mean(np.abs(errors)),
        "Bias": np.mean(errors)
    }


def make_monthly(data):
    if data.empty:
        return pd.DataFrame()

    monthly = data.copy()

    monthly["Month"] = (
        monthly["Date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    return monthly.groupby(
        ["Month", "Region"],
        as_index=False
    ).agg(
        Sales=("Sales", "sum"),
        Avg_Price=("Price", "mean"),
        Avg_Daily_Sales=("Sales", "mean")
    )


def add_intervention_line(fig):
    fig.add_vline(
        x=intervention_date.timestamp() * 1000,
        line_dash="dot",
        line_color="red"
    )

    return fig


def calculate_did(data):
    panel = data.copy()

    panel["Period"] = np.where(
        panel["Date"] >= intervention_date,
        "Post",
        "Pre"
    )

    means = panel.groupby(
        ["Region", "Period"]
    )["Sales"].mean().unstack()

    if not (
        {"Bengaluru", "Hyderabad"}.issubset(means.index)
        and {"Pre", "Post"}.issubset(means.columns)
    ):
        return None

    required = means.loc[
        ["Bengaluru", "Hyderabad"],
        ["Pre", "Post"]
    ]

    if required.isna().any().any():
        return None

    treated_change = (
        means.loc["Bengaluru", "Post"]
        - means.loc["Bengaluru", "Pre"]
    )

    control_change = (
        means.loc["Hyderabad", "Post"]
        - means.loc["Hyderabad", "Pre"]
    )

    return {
        "Treated Change": treated_change,
        "Control Change": control_change,
        "DiD Effect": treated_change - control_change
    }


# ============================================================
# MAIN HEADER
# ============================================================

st.title("📊 Algorithmic Pricing Analysis")

st.caption(
    "Causal Inference & Synthetic Control Engine"
)

st.info(
    "This project combines pricing analysis with a "
    "simulated regional experiment. The causal estimates "
    "are model-based and should not be interpreted as "
    "verified real-world business outcomes."
)


# ============================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.header("Executive Overview")

    total_sales = filtered["Sales"].sum()

    avg_price = (
        filtered["Price"].mean()
        if not filtered.empty
        else np.nan
    )

    avg_daily = (
        filtered.groupby("Date")["Sales"].sum().mean()
        if not filtered.empty
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Sales Volume",
        f"{total_sales:,.0f} units"
    )

    c2.metric(
        "Average Price",
        f"{avg_price:.2f}"
        if pd.notna(avg_price)
        else "—"
    )

    c3.metric(
        "Average Daily Sales",
        f"{avg_daily:.2f} units"
    )

    c4.metric(
        "Regions",
        filtered["Region"].nunique()
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Regional Sales Trends")

        if not filtered.empty:

            fig = px.line(
                filtered,
                x="Date",
                y="Sales",
                color="Region",
                title="Daily Sales by Region"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.warning("No data matches the filters.")

    with right:

        st.subheader("Sales Distribution")

        if not filtered.empty:

            summary = filtered.groupby(
                "Region",
                as_index=False
            )["Sales"].sum()

            fig = px.bar(
                summary,
                x="Region",
                y="Sales",
                color="Region",
                title="Total Sales by Region"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.divider()

    st.subheader("Synthetic Control Summary")

    if not post_sc.empty:

        avg_effect = post_sc["Effect"].mean()
        total_effect = post_sc["Effect"].sum()

        a, b, c = st.columns(3)

        a.metric(
            "Average Daily Uplift",
            f"{avg_effect:+.2f} units"
        )

        b.metric(
            "Total Additional Sales",
            f"{total_effect:+.2f} units"
        )

        c.metric(
            "Post-Intervention Days",
            len(post_sc)
        )

    else:
        st.info("No post-intervention observations selected.")


# ============================================================
# PAGE 2: EXPLORATORY DATA ANALYSIS
# ============================================================

elif page == "Exploratory Analysis":

    st.header("Exploratory Data Analysis")

    st.write(
        "This dashboard explores regional sales patterns "
        "from the simulated experiment. The separate "
        "Amazon sales dataset was analyzed in the project notebook."
    )

    if filtered.empty:

        st.warning("No data available for selected filters.")

    else:

        c1, c2, c3 = st.columns(3)

        c1.metric("Records", len(filtered))
        c2.metric("Regions", filtered["Region"].nunique())
        c3.metric("Days", filtered["Date"].nunique())

        st.subheader("Daily Sales Trends")

        fig = px.line(
            filtered,
            x="Date",
            y="Sales",
            color="Region",
            title="Regional Sales Over Time"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Average Price by Region")

        price_summary = filtered.groupby(
            "Region",
            as_index=False
        )["Price"].mean()

        fig = px.bar(
            price_summary,
            x="Region",
            y="Price",
            color="Region",
            text_auto=".2f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Price vs Sales")

        fig = px.scatter(
            filtered,
            x="Price",
            y="Sales",
            color="Region",
            hover_data=["Date"],
            title="Relationship Between Price and Sales"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        correlation = (
            filtered[["Price", "Sales"]]
            .corr()
            .iloc[0, 1]
        )

        st.metric(
            "Price-Sales Correlation",
            f"{correlation:.3f}"
            if pd.notna(correlation)
            else "—"
        )

        st.caption(
            "Correlation describes association, "
            "not necessarily causation."
        )

        st.subheader("Monthly Sales Summary")

        monthly = make_monthly(filtered)

        st.dataframe(
            monthly,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 3: CAUSAL IMPACT
# ============================================================

elif page == "Causal Impact":

    st.header("Causal Impact Analysis")

    st.write(
        "Synthetic Control estimates what Bengaluru sales "
        "might have been without the simulated price intervention."
    )

    if sc_view.empty:

        st.warning("No observations available.")

    else:

        st.subheader("Actual vs Synthetic Bengaluru")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=sc_view["Date"],
                y=sc_view["Bengaluru"],
                mode="lines",
                name="Actual Bengaluru",
                line=dict(
                    color="#2563eb",
                    width=3
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=sc_view["Date"],
                y=sc_view["Synthetic_Bengaluru"],
                mode="lines",
                name="Synthetic Bengaluru",
                line=dict(
                    color="#f97316",
                    width=3,
                    dash="dash"
                )
            )
        )

        add_intervention_line(fig)

        fig.update_layout(
            title="Actual vs Counterfactual Sales",
            xaxis_title="Date",
            yaxis_title="Daily Sales (Units)",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Treatment Effect")

        if not post_sc.empty:

            average_effect = post_sc["Effect"].mean()
            total_effect = post_sc["Effect"].sum()

            uplift_pct = (
                average_effect
                / post_sc["Synthetic_Bengaluru"].mean()
                * 100
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Average Daily Effect",
                f"{average_effect:+.2f}"
            )

            c2.metric(
                "Total Estimated Effect",
                f"{total_effect:+.2f}"
            )

            c3.metric(
                "Estimated Uplift",
                f"{uplift_pct:+.2f}%"
            )

            fig = px.bar(
                post_sc,
                x="Date",
                y="Effect",
                title="Daily Treatment Effect"
            )

            fig.add_hline(
                y=average_effect,
                line_dash="dash",
                line_color="orange",
                annotation_text="Average Effect"
            )

            fig.add_hline(
                y=0,
                line_color="gray"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info("No post-intervention data selected.")

        st.subheader("Difference-in-Differences")

        regional_period = df[
            (df["Date"] >= start_date)
            & (df["Date"] <= end_date)
        ]

        did = calculate_did(regional_period)

        if did is not None:

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Bengaluru Change",
                f"{did['Treated Change']:+.2f}"
            )

            c2.metric(
                "Hyderabad Change",
                f"{did['Control Change']:+.2f}"
            )

            c3.metric(
                "DiD Estimate",
                f"{did['DiD Effect']:+.2f}"
            )

        else:
            st.info(
                "Select a period containing observations "
                "before and after the intervention."
            )

        st.subheader("Treatment Effect Data")

        st.dataframe(
            sc_view,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Impact Data",
            sc_view.to_csv(index=False),
            file_name="causal_impact.csv",
            mime="text/csv"
        )


# ============================================================
# PAGE 4: MODEL VALIDATION
# ============================================================

elif page == "Model Diagnostics":

    st.header("Model Validation & Diagnostics")

    st.write(
        "This section evaluates model accuracy, "
        "compares causal estimation methods, "
        "and examines placebo and sensitivity results."
    )

    if sc_view.empty:

        st.warning("No observations available.")

    else:

        # ----------------------------------------------------
        # 1. MODEL ACCURACY
        # ----------------------------------------------------

        st.subheader("1. Model Accuracy")

        pre_metrics = calculate_errors(pre_sc)
        post_metrics = calculate_errors(post_sc)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Pre-Intervention RMSE",
            f"{pre_metrics['RMSE']:.2f}"
            if not pre_sc.empty
            else "—"
        )

        c2.metric(
            "Pre-Intervention MAE",
            f"{pre_metrics['MAE']:.2f}"
            if not pre_sc.empty
            else "—"
        )

        c3.metric(
            "Pre-Intervention Bias",
            f"{pre_metrics['Bias']:+.2f}"
            if not pre_sc.empty
            else "—"
        )

        accuracy_table = pd.DataFrame([
            {
                "Period": "Pre-Intervention",
                **pre_metrics
            },
            {
                "Period": "Post-Intervention",
                **post_metrics
            }
        ])

        st.dataframe(
            accuracy_table,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # 2. ACTUAL VS SYNTHETIC
        # ----------------------------------------------------

        st.subheader("2. Actual vs Synthetic Sales")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=sc_view["Date"],
                y=sc_view["Bengaluru"],
                name="Actual Bengaluru",
                mode="lines",
                line=dict(color="#2563eb", width=3)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=sc_view["Date"],
                y=sc_view["Synthetic_Bengaluru"],
                name="Synthetic Bengaluru",
                mode="lines",
                line=dict(
                    color="#f97316",
                    width=3,
                    dash="dash"
                )
            )
        )

        add_intervention_line(fig)

        fig.update_layout(
            title="Actual vs Synthetic Bengaluru",
            xaxis_title="Date",
            yaxis_title="Daily Sales (Units)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # 3. DID VS SYNTHETIC CONTROL
        # ----------------------------------------------------

        st.subheader("3. DiD vs Synthetic Control")

        regional_period = df[
            (df["Date"] >= start_date)
            & (df["Date"] <= end_date)
        ]

        did = calculate_did(regional_period)

        if did is not None and not post_sc.empty:

            comparison = pd.DataFrame({
                "Method": [
                    "Difference-in-Differences",
                    "Synthetic Control"
                ],
                "Estimated Effect": [
                    did["DiD Effect"],
                    post_sc["Effect"].mean()
                ]
            })

            fig = px.bar(
                comparison,
                x="Method",
                y="Estimated Effect",
                color="Method",
                text_auto=".2f",
                title="Estimated Daily Sales Effect"
            )

            fig.update_layout(
                showlegend=False,
                yaxis_title="Additional Units per Day"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "DiD compares Bengaluru with Hyderabad. "
                "Synthetic Control uses the saved model output. "
                "Both estimates depend on causal assumptions."
            )

        else:
            st.info(
                "A pre/post period is required "
                "for method comparison."
            )

        # ----------------------------------------------------
        # 4. TEMPORAL PLACEBO TESTS
        # ----------------------------------------------------

        st.subheader("4. Temporal Placebo Tests")

        st.write(
            "Fake intervention dates are introduced "
            "within the original pre-intervention period."
        )

        fake_dates = pd.to_datetime([
            "2024-01-08",
            "2024-01-12",
            "2024-01-16",
            "2024-01-20",
            "2024-01-24"
        ])

        # Use the original intervention date for placebo checks.
        original_date = pd.Timestamp("2024-01-31")

        original_pre = sc[
            sc["Date"] < original_date
        ].copy()

        original_pre["Effect"] = (
            original_pre["Bengaluru"]
            - original_pre["Synthetic_Bengaluru"]
        )

        placebo_results = []

        for fake_date in fake_dates:

            before = original_pre[
                original_pre["Date"] < fake_date
            ]["Effect"]

            after = original_pre[
                original_pre["Date"] >= fake_date
            ]["Effect"]

            if len(before) >= 2 and len(after) >= 2:

                placebo_effect = (
                    after.mean() - before.mean()
                )

                placebo_results.append({
                    "Fake Date": fake_date.strftime("%b %d"),
                    "Placebo Effect": placebo_effect
                })

        if placebo_results:

            placebo_df = pd.DataFrame(placebo_results)

            fig = px.bar(
                placebo_df,
                x="Fake Date",
                y="Placebo Effect",
                text_auto=".2f",
                title="Temporal Placebo Results"
            )

            fig.add_hline(
                y=0,
                line_color="gray"
            )

            original_post = sc[
                sc["Date"] >= original_date
            ]

            if not original_post.empty:

                original_effect = (
                    original_post["Bengaluru"]
                    - original_post["Synthetic_Bengaluru"]
                ).mean()

                fig.add_hline(
                    y=original_effect,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Actual Estimated Effect"
                )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                placebo_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "These are descriptive temporal placebo checks "
                "using the saved synthetic baseline. "
                "They are not formal significance tests."
            )

        # ----------------------------------------------------
        # 5. TRAINING WINDOW SENSITIVITY
        # ----------------------------------------------------

        st.subheader("5. Training Window Sensitivity")

        st.write(
            "The original notebook estimated treatment "
            "effects using different training-window lengths."
        )

        sensitivity = pd.DataFrame({
            "Training Days": [15, 20, 25, 30],
            "Estimated Effect": [
                21.02,
                21.24,
                20.35,
                20.33
            ]
        })

        fig = px.line(
            sensitivity,
            x="Training Days",
            y="Estimated Effect",
            markers=True,
            title="Sensitivity of Treatment Effect"
        )

        fig.add_hline(
            y=20,
            line_dash="dash",
            line_color="gray",
            annotation_text="Programmed Effect: 20"
        )

        fig.update_layout(
            xaxis_title="Training Window (Days)",
            yaxis_title="Estimated Daily Sales Effect"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            sensitivity,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Sensitivity results are previously calculated "
            "notebook values, not dynamically refitted "
            "when sidebar filters change."
        )

        st.warning(
            "All causal results are based on simulated data. "
            "Placebo tests and model fit do not establish "
            "real-world causal validity."
        )


# ============================================================
# PAGE 5: PRICING INSIGHTS
# ============================================================

elif page == "Pricing Insights":

    st.header("Pricing & Business Insights")

    st.write(
        "This section explores regional price patterns, "
        "sales performance, and estimated revenue proxies."
    )

    if filtered.empty:

        st.warning("No data available.")

    else:

        summary = filtered.groupby(
            "Region",
            as_index=False
        ).agg(
            Avg_Price=("Price", "mean"),
            Avg_Sales=("Sales", "mean"),
            Total_Sales=("Sales", "sum"),
            Observations=("Sales", "size")
        )

        summary["Revenue Proxy"] = (
            summary["Avg_Price"]
            * summary["Total_Sales"]
        )

        st.subheader("Regional Pricing Scorecard")

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        left, right = st.columns(2)

        with left:

            fig = px.scatter(
                summary,
                x="Avg_Price",
                y="Avg_Sales",
                size="Total_Sales",
                color="Region",
                hover_name="Region",
                title="Average Price vs Sales"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:

            fig = px.bar(
                summary,
                x="Region",
                y="Revenue Proxy",
                color="Region",
                title="Revenue Proxy by Region"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("Bengaluru Before vs After")

        bengaluru = df[
            df["Region"] == "Bengaluru"
        ].copy()

        bengaluru["Period"] = np.where(
            bengaluru["Date"] >= intervention_date,
            "Post",
            "Pre"
        )

        comparison = bengaluru.groupby(
            "Period"
        ).agg(
            Avg_Price=("Price", "mean"),
            Avg_Sales=("Sales", "mean")
        )

        if {"Pre", "Post"}.issubset(comparison.index):

            price_change = (
                comparison.loc["Post", "Avg_Price"]
                / comparison.loc["Pre", "Avg_Price"]
                - 1
            ) * 100

            sales_change = (
                comparison.loc["Post", "Avg_Sales"]
                / comparison.loc["Pre", "Avg_Sales"]
                - 1
            ) * 100

            c1, c2 = st.columns(2)

            c1.metric(
                "Price Change",
                f"{price_change:+.2f}%"
            )

            c2.metric(
                "Sales Change",
                f"{sales_change:+.2f}%"
            )

        st.caption(
            "Revenue proxy = average price multiplied "
            "by total sales units. It is not actual "
            "transaction-level revenue. Before/after "
            "changes are descriptive, not causal."
        )


# ============================================================
# PAGE 6: DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.header("Data Explorer")

    tab1, tab2 = st.tabs([
        "Regional Sales",
        "Synthetic Control"
    ])

    with tab1:

        st.subheader("Regional Sales Dataset")

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Regional Data",
            filtered.to_csv(index=False),
            file_name="regional_sales_filtered.csv",
            mime="text/csv"
        )

    with tab2:

        st.subheader("Synthetic Control Dataset")

        st.dataframe(
            sc_view,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Synthetic Data",
            sc_view.to_csv(index=False),
            file_name="synthetic_control_filtered.csv",
            mime="text/csv"
        )

    with st.expander("Data Dictionary"):

        st.markdown("""
        - **Date:** Observation date.
        - **Region:** Geographic market.
        - **Price:** Simulated regional price.
        - **Sales:** Daily sales units.
        - **Bengaluru:** Actual simulated Bengaluru sales.
        - **Synthetic_Bengaluru:** Estimated counterfactual sales.
        - **Effect:** Actual sales minus synthetic sales.
        """)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Algorithmic Pricing Analysis | "
    "Data Analytics Project | "
    "Built with Python, Streamlit and Plotly | "
    "Simulated Data"
)
