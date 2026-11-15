from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# Configuration and data
# ------------------------------------------------------------
st.set_page_config(page_title="Algorithmic Pricing Analytics", page_icon="📊", layout="wide")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_INTERVENTION = pd.Timestamp("2024-01-31")

@st.cache_data
def load_data():
    regional = pd.read_csv(DATA_DIR / "regional_sales_simulated.csv")
    synthetic = pd.read_csv(DATA_DIR / "synthetic_control_results.csv")
    regional["Date"] = pd.to_datetime(regional["Date"], errors="coerce")
    synthetic["Date"] = pd.to_datetime(synthetic["Date"], errors="coerce")
    regional = regional.dropna(subset=["Date", "Region", "Price", "Sales"]).copy()
    synthetic = synthetic.dropna(subset=["Date", "Bengaluru", "Synthetic_Bengaluru"]).copy()
    return regional, synthetic

try:
    df, sc = load_data()
except Exception as exc:
    st.error(f"Could not load project datasets: {exc}")
    st.stop()

required_regional = {"Date", "Region", "Price", "Sales"}
required_sc = {"Date", "Bengaluru", "Synthetic_Bengaluru"}
if not required_regional.issubset(df.columns) or not required_sc.issubset(sc.columns):
    st.error("The CSV schema does not match the expected project datasets.")
    st.stop()

regions = sorted(df["Region"].unique().tolist())
min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
if DEFAULT_INTERVENTION > df["Date"].max():
    DEFAULT_INTERVENTION = df["Date"].median().normalize()

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("📊 Pricing Analytics")
page = st.sidebar.radio(
    "Workspace",
    ["Executive Overview", "Exploratory Analysis", "Causal Impact",
     "Model Diagnostics", "Pricing Insights", "Data Explorer"]
)
st.sidebar.divider()
st.sidebar.caption("Global filters")
date_range = st.sidebar.date_input("Analysis period", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_date, end_date = pd.Timestamp(min_date), pd.Timestamp(max_date)
selected_regions = st.sidebar.multiselect("Regions", regions, default=regions)
filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date) &
              (df["Region"].isin(selected_regions))].copy()
intervention_date = st.sidebar.date_input(
    "Price intervention date", value=DEFAULT_INTERVENTION.date(),
    min_value=min_date, max_value=max_date
)
intervention_date = pd.Timestamp(intervention_date)
st.sidebar.caption("Intervention assumption: Bengaluru price reduction from ₹100 to ₹90.")
st.sidebar.caption("All project datasets are simulated.")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def metric_delta(current, previous, suffix=""):
    if pd.isna(previous) or previous == 0:
        return "n/a"
    return f"{(current / previous - 1) * 100:+.1f}%{suffix}"

def empty_warning(message):
    st.warning(message)

def make_monthly(data):
    if data.empty:
        return data
    return (data.assign(Month=data["Date"].dt.to_period("M").dt.to_timestamp())
            .groupby(["Month", "Region"], as_index=False)
            .agg(Sales=("Sales", "sum"), Avg_Price=("Price", "mean"),
                 Avg_Daily_Sales=("Sales", "mean")))

# Synthetic-control outcomes
sc_view = sc[(sc["Date"] >= start_date) & (sc["Date"] <= end_date)].copy()
sc_view["Effect"] = sc_view["Bengaluru"] - sc_view["Synthetic_Bengaluru"]
post_sc = sc_view[sc_view["Date"] >= intervention_date]
pre_sc = sc_view[sc_view["Date"] < intervention_date]

st.title("Algorithmic Pricing Analysis")
st.caption("Regional sales • Price-performance analysis • Synthetic control evaluation")

# ------------------------------------------------------------
# Executive Overview
# ------------------------------------------------------------
if page == "Executive Overview":
    st.subheader("Executive overview")
    total_sales = filtered["Sales"].sum() if not filtered.empty else 0
    avg_price = filtered["Price"].mean() if not filtered.empty else np.nan
    avg_daily = filtered.groupby("Date")["Sales"].sum().mean() if not filtered.empty else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sales volume", f"{total_sales:,.0f} units")
    c2.metric("Average selling price", f"₹{avg_price:,.2f}" if pd.notna(avg_price) else "—")
    c3.metric("Average daily sales", f"{avg_daily:,.1f} units/day")
    c4.metric("Regions in view", f"{filtered['Region'].nunique() if not filtered.empty else 0}")
    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("#### Daily sales trend")
        if not filtered.empty:
            daily = filtered.groupby(["Date", "Region"], as_index=False)["Sales"].sum()
            fig = px.line(daily, x="Date", y="Sales", color="Region", markers=True)
            fig.update_layout(xaxis_title="", yaxis_title="Units sold", legend_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty_warning("No rows match the selected filters.")
    with right:
        st.markdown("#### Sales by region")
        if not filtered.empty:
            summary = filtered.groupby("Region", as_index=False).agg(
                Sales=("Sales", "sum"), Avg_Price=("Price", "mean"))
            fig = px.bar(summary.sort_values("Sales", ascending=True), x="Sales", y="Region",
                         orientation="h", color="Region", text_auto=".3s")
            fig.update_layout(showlegend=False, xaxis_title="Total units sold", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Synthetic-control headline result")
    if not post_sc.empty:
        avg_effect = post_sc["Effect"].mean()
        total_effect = post_sc["Effect"].sum()
        a, b, c = st.columns(3)
        a.metric("Average post-period gap", f"{avg_effect:+.2f} units/day")
        b.metric("Cumulative post-period gap", f"{total_effect:+.1f} units")
        c.metric("Post-period days", f"{len(post_sc)}")
        st.caption("Gap = actual Bengaluru sales − synthetic Bengaluru sales. This is a model-based estimate on simulated data, not proof of causality.")
    else:
        st.info("The selected period contains no post-intervention dates.")

# ------------------------------------------------------------
# EDA
# ------------------------------------------------------------
elif page == "Exploratory Analysis":
    st.subheader("Exploratory data analysis")
    if filtered.empty:
        empty_warning("No data available for the selected filters.")
    else:
        a, b, c = st.columns(3)
        a.metric("Rows", f"{len(filtered):,}")
        b.metric("Unique dates", f"{filtered['Date'].nunique():,}")
        c.metric("Products / regions", f"{filtered['Region'].nunique():,} regions")
        monthly = make_monthly(filtered)
        l, r = st.columns(2)
        with l:
            st.markdown("#### Daily sales by region")
            daily = filtered.groupby(["Date", "Region"], as_index=False)["Sales"].sum()
            fig = px.line(daily, x="Date", y="Sales", color="Region")
            st.plotly_chart(fig, use_container_width=True)
        with r:
            st.markdown("#### Average price by region")
            p = filtered.groupby("Region", as_index=False)["Price"].mean()
            fig = px.bar(p, x="Region", y="Price", color="Region", text_auto=".2f")
            fig.update_layout(showlegend=False, yaxis_title="Average price")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Price versus sales")
        fig = px.scatter(filtered, x="Price", y="Sales", color="Region",
                         trendline="ols", hover_data=["Date"])
        st.plotly_chart(fig, use_container_width=True)
        corr = filtered[["Price", "Sales"]].corr().iloc[0, 1]
        st.metric("Pooled price–sales correlation", f"{corr:.3f}" if pd.notna(corr) else "—")
        st.caption("Correlation is descriptive. It does not isolate price effects from time, region, demand, or other confounders.")
        st.markdown("#### Monthly summary")
        st.dataframe(monthly, use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# Causal impact
# ------------------------------------------------------------
elif page == "Causal Impact":
    st.subheader("Synthetic-control impact analysis")
    st.write("The synthetic series is the project's counterfactual benchmark for Bengaluru. The intervention date is configurable in the sidebar.")
    if sc_view.empty:
        empty_warning("No synthetic-control observations match the selected date range.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sc_view["Date"], y=sc_view["Bengaluru"], mode="lines",
                                 name="Actual Bengaluru", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=sc_view["Date"], y=sc_view["Synthetic_Bengaluru"], mode="lines",
                                 name="Synthetic Bengaluru", line=dict(width=3, dash="dash")))
        fig.add_vline(x=intervention_date, line_dash="dot", line_color="red",
                      annotation_text="Intervention", annotation_position="top")
        fig.update_layout(xaxis_title="", yaxis_title="Daily sales (units)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        if not post_sc.empty:
            x1, x2, x3 = st.columns(3)
            x1.metric("Mean daily estimated effect", f"{post_sc['Effect'].mean():+.2f} units")
            x2.metric("Cumulative estimated effect", f"{post_sc['Effect'].sum():+.1f} units")
            x3.metric("Mean uplift vs synthetic", f"{(post_sc['Effect'].mean()/post_sc['Synthetic_Bengaluru'].mean())*100:+.2f}%")
            fig2 = px.bar(post_sc, x="Date", y="Effect", title="Daily treatment-effect gap")
            fig2.add_hline(y=post_sc["Effect"].mean(), line_dash="dash",
                           annotation_text=f"Mean: {post_sc['Effect'].mean():.2f}")
            fig2.add_hline(y=0, line_color="gray")
            fig2.update_layout(xaxis_title="", yaxis_title="Actual − synthetic (units)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No post-intervention observations in the selected date range.")
        st.markdown("#### Difference-in-differences cross-check")
        regional_period = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()
        regional_period["Period"] = np.where(regional_period["Date"] >= intervention_date, "Post", "Pre")
        means = regional_period[regional_period["Region"].isin(regions)].groupby(
            ["Region", "Period"], as_index=False)["Sales"].mean()
        pivot = means.pivot(index="Region", columns="Period", values="Sales")
        if "Pre" in pivot.columns and "Post" in pivot.columns and "Bengaluru" in pivot.index:
            controls = [r for r in pivot.index if r != "Bengaluru"]
            if controls and pivot.loc[["Bengaluru"] + controls, ["Pre", "Post"]].notna().all().all():
                control_change = (pivot.loc[controls, "Post"] - pivot.loc[controls, "Pre"]).mean()
                treated_change = pivot.loc["Bengaluru", "Post"] - pivot.loc["Bengaluru", "Pre"]
                did = treated_change - control_change
                q1, q2, q3 = st.columns(3)
                q1.metric("Bengaluru pre/post change", f"{treated_change:+.2f}")
                q2.metric("Average control change", f"{control_change:+.2f}")
                q3.metric("DiD estimate", f"{did:+.2f} units/day")
                st.caption("DiD = (Bengaluru post − pre) − (average control post − pre). Interpret cautiously: parallel trends and no spillovers are not established by this small simulated dataset.")
            else:
                st.info("Insufficient pre/post observations across treated and control regions for DiD.")
        else:
            st.info("Choose a date range spanning both sides of the intervention to calculate DiD.")
        st.markdown("#### Effect data")
        st.dataframe(sc_view[["Date", "Bengaluru", "Synthetic_Bengaluru", "Effect"]],
                     use_container_width=True, hide_index=True)
        st.download_button("Download impact data", sc_view.to_csv(index=False).encode("utf-8"),
                           "synthetic_control_impact.csv", "text/csv")

# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------
elif page == "Model Diagnostics":
    st.subheader("Model validation and diagnostics")
    st.caption("All results use simulated data. The default intervention date is January 31, 2024.")
    if sc_view.empty:
        empty_warning("No observations in the selected date range.")
    else:
        pre = sc_view[sc_view["Date"] < intervention_date].copy()
        post = sc_view[sc_view["Date"] >= intervention_date].copy()

        def errors(part):
            residual = part["Effect"]
            return {
                "Observations": len(residual),
                "MAE": residual.abs().mean(),
                "RMSE": np.sqrt((residual ** 2).mean()),
                "Mean residual": residual.mean(),
            }

        pre_m, post_m = errors(pre), errors(post)
        cols = st.columns(4)
        cols[0].metric("Pre-period RMSE", f"{pre_m['RMSE']:.2f}" if len(pre) else "—")
        cols[1].metric("Pre-period MAE", f"{pre_m['MAE']:.2f}" if len(pre) else "—")
        cols[2].metric("Post-period RMSE", f"{post_m['RMSE']:.2f}" if len(post) else "—")
        cols[3].metric("Post-period MAE", f"{post_m['MAE']:.2f}" if len(post) else "—")

        st.markdown("#### Actual versus synthetic: model fit")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sc_view["Date"], y=sc_view["Bengaluru"],
                                 name="Actual Bengaluru", mode="lines"))
        fig.add_trace(go.Scatter(x=sc_view["Date"], y=sc_view["Synthetic_Bengaluru"],
                                 name="Synthetic Bengaluru", mode="lines",
                                 line=dict(dash="dash")))
        fig.add_vline(x=intervention_date.timestamp() * 1000,
                      line_dash="dot", line_color="red")
        fig.update_layout(xaxis_title="Date", yaxis_title="Daily sales (units)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Residuals: actual minus synthetic")
        residual_fig = px.line(sc_view, x="Date", y="Effect", markers=True)
        residual_fig.add_hline(y=0, line_color="gray")
        residual_fig.add_vline(x=intervention_date.timestamp() * 1000,
                               line_dash="dot", line_color="red")
        residual_fig.update_layout(yaxis_title="Sales gap (units)")
        st.plotly_chart(residual_fig, use_container_width=True)
        st.dataframe(pd.DataFrame([{"Period": "Before intervention", **pre_m},
                                   {"Period": "After intervention", **post_m}]),
                     hide_index=True, use_container_width=True)

        st.markdown("#### Difference-in-Differences versus Synthetic Control")
        regional_period = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()
        regional_period["Period"] = np.where(regional_period["Date"] >= intervention_date,
                                             "Post", "Pre")
        means = regional_period.groupby(["Region", "Period"])["Sales"].mean().unstack()
        if ("Bengaluru" in means.index and {"Pre", "Post"}.issubset(means.columns)
                and "Hyderabad" in means.index
                and means.loc[["Bengaluru", "Hyderabad"], ["Pre", "Post"]].notna().all().all()
                and not post.empty):
            treated_change = means.loc["Bengaluru", "Post"] - means.loc["Bengaluru", "Pre"]
            control_change = means.loc["Hyderabad", "Post"] - means.loc["Hyderabad", "Pre"]
            did_effect = treated_change - control_change
            synthetic_effect = post["Effect"].mean()
            method_df = pd.DataFrame({"Method": ["DiD (Hyderabad control)", "Synthetic Control"],
                                      "Estimated uplift": [did_effect, synthetic_effect]})
            compare_fig = px.bar(method_df, x="Method", y="Estimated uplift",
                                 color="Method", text_auto=".2f")
            compare_fig.update_layout(showlegend=False, yaxis_title="Additional units/day")
            st.plotly_chart(compare_fig, use_container_width=True)
            st.caption("DiD uses Hyderabad as its comparison region. Synthetic Control uses the saved counterfactual series. Neither estimate establishes causality without assumptions.")
        else:
            st.info("Select a date range with pre- and post-intervention data to compare DiD and Synthetic Control.")

        st.markdown("#### Temporal placebo checks")
        st.write("Fake intervention dates within the real pre-intervention period test whether similar gaps appear without the programmed price change.")
        fake_dates = pd.to_datetime(["2024-01-08", "2024-01-12", "2024-01-16",
                                     "2024-01-20", "2024-01-24"])
        placebo_rows = []
        # Use the fixed, saved counterfactual series; no model is refitted here.
        for fake_date in fake_dates:
            before = pre.loc[pre["Date"] < fake_date, "Effect"]
            after = pre.loc[pre["Date"] >= fake_date, "Effect"]
            if len(before) >= 2 and len(after) >= 2:
                placebo_rows.append({"Fake date": fake_date.strftime("%b %d"),
                                     "Gap change": after.mean() - before.mean()})
        if placebo_rows:
            placebo_df = pd.DataFrame(placebo_rows)
            placebo_fig = px.bar(placebo_df, x="Fake date", y="Gap change",
                                 text_auto=".2f")
            placebo_fig.add_hline(y=0, line_color="gray")
            if not post.empty:
                placebo_fig.add_hline(y=post["Effect"].mean(), line_dash="dash",
                                      line_color="green",
                                      annotation_text="Actual post-period mean gap")
            placebo_fig.update_layout(yaxis_title="Change in mean residual (units/day)")
            st.plotly_chart(placebo_fig, use_container_width=True)
            st.caption("Temporal placebos are descriptive checks using the existing fitted synthetic series, not formal significance tests.")
        else:
            st.info("The selected pre-intervention period is too short for the fake-date checks.")

        st.markdown("#### Training-window sensitivity (notebook results)")
        st.write("These are previously reported results from the notebook, not recalculated when sidebar filters change.")
        sensitivity = pd.DataFrame({"Training days": [15, 20, 25, 30],
                                    "Estimated uplift": [21.02, 21.24, 20.35, 20.33]})
        sensitivity_fig = px.line(sensitivity, x="Training days", y="Estimated uplift",
                                  markers=True)
        sensitivity_fig.add_hline(y=20, line_dash="dash", line_color="gray",
                                  annotation_text="Programmed effect: 20")
        sensitivity_fig.update_layout(yaxis_title="Estimated additional units/day")
        st.plotly_chart(sensitivity_fig, use_container_width=True)
        st.dataframe(sensitivity, hide_index=True, use_container_width=True)
        st.warning("Validation checks support the simulated model's plausibility but do not prove a causal effect in real-world sales data.")

# ------------------------------------------------------------
# Pricing insights
# ------------------------------------------------------------
elif page == "Pricing Insights":
    st.subheader("Pricing and commercial insights")
    st.write("These are descriptive signals from the supplied regional panel and synthetic-control output.")
    if filtered.empty:
        empty_warning("No data available for the selected filters.")
    else:
        summary = filtered.groupby("Region", as_index=False).agg(
            Avg_Price=("Price", "mean"), Avg_Sales=("Sales", "mean"),
            Total_Sales=("Sales", "sum"), Observations=("Sales", "size"))
        summary["Revenue proxy"] = summary["Avg_Price"] * summary["Total_Sales"]
        st.markdown("#### Regional price and sales scorecard")
        st.dataframe(summary.sort_values("Total_Sales", ascending=False),
                     hide_index=True, use_container_width=True)
        l, r = st.columns(2)
        with l:
            fig = px.scatter(summary, x="Avg_Price", y="Avg_Sales", size="Total_Sales",
                             color="Region", text="Region", title="Average price vs average daily sales")
            st.plotly_chart(fig, use_container_width=True)
        with r:
            fig = px.bar(summary, x="Region", y="Revenue proxy", color="Region",
                         title="Revenue proxy by region")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        beng = df[df["Region"] == "Bengaluru"].copy()
        beng["Period"] = np.where(beng["Date"] >= intervention_date, "Post", "Pre")
        bp = beng.groupby("Period").agg(Avg_Price=("Price","mean"), Avg_Sales=("Sales","mean"))
        if {"Pre","Post"}.issubset(bp.index):
            price_change = (bp.loc["Post","Avg_Price"] / bp.loc["Pre","Avg_Price"] - 1) * 100
            sales_change = (bp.loc["Post","Avg_Sales"] / bp.loc["Pre","Avg_Sales"] - 1) * 100
            st.markdown("#### Bengaluru before/after comparison")
            x, y = st.columns(2)
            x.metric("Average price change", f"{price_change:+.1f}%")
            y.metric("Average sales change", f"{sales_change:+.1f}%")
        st.caption("Revenue proxy = average price × total units; it is not transaction-level revenue. Before/after changes are not causal estimates.")

    st.divider()
    st.markdown('#### Synthetic-control pricing impact')
    st.caption('Calculated from the saved counterfactual series for the selected post-intervention dates.')
    if not post_sc.empty:
        expected_sales = post_sc['Synthetic_Bengaluru'].mean()
        observed_sales = post_sc['Bengaluru'].mean()
        effect = observed_sales - expected_sales
        uplift = 100 * effect / expected_sales if expected_sales else np.nan
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Expected daily sales', f'{expected_sales:.2f}')
        c2.metric('Observed daily sales', f'{observed_sales:.2f}')
        c3.metric('Additional units/day', f'{effect:+.2f}')
        c4.metric('Estimated uplift', f'{uplift:+.2f}%')
        chart = pd.DataFrame({'Scenario': ['Expected without change', 'Observed after change'],
                              'Daily sales': [expected_sales, observed_sales]})
        st.plotly_chart(px.bar(chart, x='Scenario', y='Daily sales', color='Scenario',
                               text_auto='.2f', title='Expected vs observed daily sales'),
                        use_container_width=True)
        pre_prices = df[(df['Region'] == 'Bengaluru') & (df['Date'] < DEFAULT_INTERVENTION)]['Price']
        post_prices = df[(df['Region'] == 'Bengaluru') & (df['Date'] >= DEFAULT_INTERVENTION)]['Price']
        if (intervention_date == DEFAULT_INTERVENTION and
                start_date <= DEFAULT_INTERVENTION - pd.Timedelta(days=1) and
                end_date >= DEFAULT_INTERVENTION and
                not pre_prices.empty and not post_prices.empty):
            pct_price = 100 * (post_prices.mean() / pre_prices.mean() - 1)
            if pct_price != 0 and pd.notna(uplift):
                st.markdown('#### Approximate adjusted price elasticity')
                st.metric('Adjusted elasticity', f'{uplift / pct_price:.2f}')
                st.caption('Approximation: model-estimated sales uplift (%) / price change (%). Not a separately fitted demand model.')
        else:
            st.info('For the original price elasticity comparison, select January 31, 2024 as the intervention and a date range covering before and after.')
        st.markdown('#### Interpretation')
        st.write('In this simulated experiment, the lower price is associated with higher estimated unit demand. This does not establish that revenue or profit improved; costs and margins must be evaluated separately.')
    else:
        st.info('Select dates after the intervention to view pricing impact.')

# ------------------------------------------------------------
# Data Explorer
# ------------------------------------------------------------
elif page == "Data Explorer":
    st.subheader("Data explorer")
    tab1, tab2 = st.tabs(["Regional sales data", "Synthetic-control data"])
    with tab1:
        st.write(f"{len(filtered):,} rows after applying global filters.")
        st.dataframe(filtered.sort_values("Date"), use_container_width=True, hide_index=True)
        st.download_button("Download filtered regional data",
                           filtered.to_csv(index=False).encode("utf-8"),
                           "regional_sales_filtered.csv", "text/csv")
    with tab2:
        st.dataframe(sc_view.sort_values("Date"), use_container_width=True, hide_index=True)
        st.download_button("Download filtered synthetic data",
                           sc_view.to_csv(index=False).encode("utf-8"),
                           "synthetic_control_filtered.csv", "text/csv")
    with st.expander("Data dictionary"):
        st.markdown("""
        - **Date:** observation date.
        - **Region:** geographic sales market.
        - **Price:** observed price in the supplied dataset (currency not explicitly specified in source).
        - **Sales:** daily units sold.
        - **Bengaluru:** observed daily Bengaluru sales in the synthetic-control dataset.
        - **Synthetic_Bengaluru:** model-generated counterfactual benchmark for Bengaluru.
        """)

st.divider()
st.caption("Algorithmic Pricing Analysis | Built with Streamlit and Plotly | Source: supplied simulated project datasets")
