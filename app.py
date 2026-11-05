from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from analytics import load_data as read_data, calculate_errors, calculate_did, monthly_summary

st.set_page_config(page_title="Algorithmic Pricing Analysis", page_icon="📊", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_INTERVENTION = pd.Timestamp("2024-01-31")
REQUIRED_REGIONAL = {"Date", "Region", "Price", "Sales"}
REQUIRED_SYNTHETIC = {"Date", "Bengaluru", "Synthetic_Bengaluru"}


@st.cache_data(show_spinner="Loading and validating project data…")
def load_data():
    return read_data(DATA_DIR)



try:
    df, sc, audit = load_data()
except Exception as exc:
    st.error(f"Data loading failed: {exc}")
    st.stop()

st.sidebar.title("📊 Pricing Analytics")
page = st.sidebar.radio("Workspace", ["Executive Overview", "Exploratory Analysis", "Causal Impact",
    "Model Diagnostics", "Pricing Insights", "Data Explorer"])
st.sidebar.divider()
st.sidebar.subheader("Global filters")
min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
raw_range = st.sidebar.date_input("Analysis period", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(raw_range, (tuple, list)):
    start_date = pd.Timestamp(raw_range[0])
    end_date = pd.Timestamp(raw_range[-1])
else:
    start_date = end_date = pd.Timestamp(raw_range)
if start_date > end_date:
    start_date, end_date = end_date, start_date
regions = sorted(df["Region"].unique().tolist())
selected_regions = st.sidebar.multiselect("Regions", regions, default=regions)
intervention_date = pd.Timestamp(st.sidebar.date_input("Intervention / split date", value=DEFAULT_INTERVENTION.date(),
    min_value=min_date, max_value=max_date, help="Changes pre/post summaries; it does not refit the saved synthetic series."))

filtered = df[df["Date"].between(start_date, end_date) & df["Region"].isin(selected_regions)].copy()
sc_view = sc[sc["Date"].between(start_date, end_date)].copy()
sc_view["Effect"] = sc_view["Bengaluru"] - sc_view["Synthetic_Bengaluru"]
pre_sc = sc_view[sc_view["Date"] < intervention_date]
post_sc = sc_view[sc_view["Date"] >= intervention_date]

st.title("📊 Algorithmic Pricing Analysis")
st.caption(f"Data-backed pricing and causal-analysis workspace · {start_date:%d %b %Y} – {end_date:%d %b %Y}")
st.info("The supplied datasets are simulated. All figures below are recalculated from the bundled CSVs; descriptive estimates are not proof of causality.")

if page == "Executive Overview":
    st.header("Executive Overview")
    if filtered.empty:
        st.warning("No regional rows match the current filters.")
    else:
        total_units = float(filtered["Sales"].sum())
        avg_price = float(filtered["Price"].mean())
        revenue_proxy = float((filtered["Price"] * filtered["Sales"]).sum())
        active_days = int(filtered["Date"].nunique())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Units sold", f"{total_units:,.0f}")
        c2.metric("Mean listed price", f"{avg_price:,.2f}")
        c3.metric("Revenue proxy", f"{revenue_proxy:,.0f}")
        c4.metric("Observed days", f"{active_days:,}")
        trend = px.line(filtered, x="Date", y="Sales", color="Region", markers=False, title="Daily sales by region")
        trend.add_vline(x=intervention_date, line_dash="dash", line_color="red", annotation_text="Split date")
        st.plotly_chart(trend, use_container_width=True)
        summary = filtered.groupby("Region", as_index=False).agg(Units=("Sales", "sum"), **{"Mean price": ("Price", "mean"), "Mean daily sales": ("Sales", "mean")})
        left, right = st.columns(2)
        left.plotly_chart(px.bar(summary, x="Region", y="Units", color="Region", title="Total units by region"), use_container_width=True)
        right.plotly_chart(px.bar(summary, x="Region", y="Mean price", color="Region", title="Mean price by region"), use_container_width=True)
    st.subheader("Synthetic-control signal")
    if post_sc.empty:
        st.warning("No post-split synthetic observations in the selected period.")
    else:
        effect = float(post_sc["Effect"].mean())
        cumulative = float(post_sc["Effect"].sum())
        baseline = float(post_sc["Synthetic_Bengaluru"].mean())
        uplift = effect / baseline * 100 if baseline else np.nan
        a, b, c = st.columns(3)
        a.metric("Mean daily gap", f"{effect:+.2f} units")
        b.metric("Sum of observed daily gaps", f"{cumulative:+.1f} units")
        c.metric("Gap vs synthetic mean", f"{uplift:+.2f}%" if pd.notna(uplift) else "N/A")
        st.caption("These are descriptive actual-minus-synthetic gaps over available post-split dates, not a confidence interval or verified causal effect.")

elif page == "Exploratory Analysis":
    st.header("Exploratory Analysis")
    if filtered.empty:
        st.warning("No observations match the current filters.")
    else:
        a, b, c = st.columns(3)
        a.metric("Rows", f"{len(filtered):,}")
        b.metric("Regions", filtered["Region"].nunique())
        c.metric("Dates", filtered["Date"].nunique())
        st.plotly_chart(px.line(filtered, x="Date", y="Sales", color="Region", title="Daily sales trajectory"), use_container_width=True)
        left, right = st.columns(2)
        left.plotly_chart(px.scatter(filtered, x="Price", y="Sales", color="Region", hover_data=["Date"], title="Price–sales observations"), use_container_width=True)
        corr = filtered[["Price", "Sales"]].corr().iloc[0, 1] if len(filtered) > 1 else np.nan
        right.metric("Pooled price–sales correlation", f"{corr:+.3f}" if pd.notna(corr) else "N/A")
        right.caption("Pooled correlation is descriptive and may reflect region/time confounding; it is not an elasticity estimate.")
        st.subheader("Monthly roll-up")
        st.dataframe(monthly_summary(filtered), use_container_width=True, hide_index=True)

elif page == "Causal Impact":
    st.header("Causal Impact")
    st.warning("Interpretation guardrail: a precomputed synthetic series and a two-region DiD are not sufficient on their own to establish causality. Validate pre-trends, donor construction, concurrent shocks and uncertainty before making a causal claim.")
    if sc_view.empty:
        st.warning("No synthetic-control rows match the selected period.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sc_view.Date, y=sc_view.Bengaluru, mode="lines", name="Observed Bengaluru"))
        fig.add_trace(go.Scatter(x=sc_view.Date, y=sc_view.Synthetic_Bengaluru, mode="lines", name="Synthetic Bengaluru", line={"dash": "dash"}))
        fig.add_vline(x=intervention_date, line_dash="dash", line_color="red", annotation_text="Split date")
        fig.update_layout(title="Observed vs synthetic Bengaluru", xaxis_title="Date", yaxis_title="Sales (units)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        if post_sc.empty:
            st.info("No post-split dates are available in this window.")
        else:
            avg_gap = float(post_sc.Effect.mean())
            sum_gap = float(post_sc.Effect.sum())
            st.metric("Mean post-split daily gap", f"{avg_gap:+.2f} units")
            st.metric("Sum of post-split observed daily gaps", f"{sum_gap:+.1f} units")
            gapfig = px.bar(post_sc, x="Date", y="Effect", title="Daily observed-minus-synthetic gap")
            gapfig.add_hline(y=0, line_color="gray")
            st.plotly_chart(gapfig, use_container_width=True)
    st.subheader("Difference-in-Differences (Bengaluru vs Hyderabad)")
    did = calculate_did(filtered, intervention_date)
    if did is None:
        st.info("DiD needs both Bengaluru and Hyderabad, with observations before and on/after the split date.")
    else:
        x, y, z = st.columns(3)
        x.metric("Bengaluru pre/post change", f"{did['Treated change']:+.2f} units/day")
        y.metric("Hyderabad pre/post change", f"{did['Control change']:+.2f} units/day")
        z.metric("Descriptive DiD contrast", f"{did['DiD estimate']:+.2f} units/day")
        st.dataframe(did["means"].rename_axis("Region").reset_index(), use_container_width=True, hide_index=True)

elif page == "Model Diagnostics":
    st.header("Model Diagnostics")
    if sc_view.empty:
        st.warning("No synthetic-control observations match the selected period.")
    else:
        pre_metrics, post_metrics = calculate_errors(pre_sc), calculate_errors(post_sc)
        st.subheader("Fit against the stored series")
        a, b, c = st.columns(3)
        a.metric("Pre-split RMSE", f"{pre_metrics['RMSE']:.3f}" if pd.notna(pre_metrics["RMSE"]) else "N/A")
        b.metric("Pre-split MAE", f"{pre_metrics['MAE']:.3f}" if pd.notna(pre_metrics["MAE"]) else "N/A")
        c.metric("Pre-split bias", f"{pre_metrics['Bias']:+.3f}" if pd.notna(pre_metrics["Bias"]) else "N/A")
        st.dataframe(pd.DataFrame([{"Window": "Pre-split", **pre_metrics}, {"Window": "Post-split", **post_metrics}]), use_container_width=True, hide_index=True)
        fit = go.Figure()
        fit.add_trace(go.Scatter(x=sc_view.Date, y=sc_view.Bengaluru, name="Observed", mode="lines"))
        fit.add_trace(go.Scatter(x=sc_view.Date, y=sc_view.Synthetic_Bengaluru, name="Synthetic", mode="lines"))
        fit.add_vline(x=intervention_date, line_dash="dash", line_color="red")
        fit.update_layout(title="Observed and synthetic series", yaxis_title="Sales (units)")
        st.plotly_chart(fit, use_container_width=True)
        st.subheader("Pre-intervention temporal placebo diagnostics")
        placebo_pool = sc_view.loc[sc_view.Date < intervention_date].copy()
        placebo_pool["Gap"] = placebo_pool.Bengaluru - placebo_pool.Synthetic_Bengaluru
        candidates = placebo_pool.Date.sort_values().iloc[2:-2: max(1, len(placebo_pool)//8)] if len(placebo_pool) >= 6 else []
        rows = []
        for fake in candidates:
            before = placebo_pool.loc[placebo_pool.Date < fake, "Gap"]
            after = placebo_pool.loc[placebo_pool.Date >= fake, "Gap"]
            if len(before) >= 2 and len(after) >= 2:
                rows.append({"Pseudo split": fake, "Pre/post mean-gap change": float(after.mean() - before.mean())})
        if rows:
            placebo = pd.DataFrame(rows)
            st.plotly_chart(px.bar(placebo, x="Pseudo split", y="Pre/post mean-gap change", title="Placebo contrasts within pre-period"), use_container_width=True)
            st.dataframe(placebo, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough pre-period observations for placebo diagnostics.")
        st.caption("No hardcoded training-window sensitivity values are shown. A true sensitivity analysis requires refitting the synthetic-control model under alternative training windows.")

elif page == "Pricing Insights":
    st.header("Pricing Insights")
    if filtered.empty:
        st.warning("No observations match the current filters.")
    else:
        x = filtered.assign(Revenue_Proxy=filtered.Price * filtered.Sales)
        summary = x.groupby("Region", as_index=False).agg(
            **{"Mean price": ("Price", "mean"), "Mean daily units": ("Sales", "mean"),
               "Total units": ("Sales", "sum"), "Revenue proxy": ("Revenue_Proxy", "sum"), "Rows": ("Sales", "size")})
        st.subheader("Regional commercial scorecard")
        st.dataframe(summary.sort_values("Revenue proxy", ascending=False), use_container_width=True, hide_index=True)
        st.plotly_chart(px.scatter(summary, x="Mean price", y="Mean daily units", size="Total units", color="Region", hover_name="Region", title="Regional mean price vs mean daily units"), use_container_width=True)
        st.subheader("Bengaluru pre/post description")
        b = filtered[filtered.Region.eq("Bengaluru")].copy()
        if b.empty:
            st.info("Select Bengaluru to calculate its pre/post descriptive comparison.")
        else:
            b["Period"] = np.where(b.Date < intervention_date, "Pre", "Post")
            comp = b.groupby("Period").agg(**{"Mean price": ("Price", "mean"), "Mean daily units": ("Sales", "mean")}).reindex(["Pre", "Post"])
            if comp.notna().all().all():
                st.dataframe(comp, use_container_width=True)
                if comp.loc["Pre", "Mean price"] != 0 and comp.loc["Pre", "Mean daily units"] != 0:
                    pchg = (comp.loc["Post", "Mean price"] / comp.loc["Pre", "Mean price"] - 1) * 100
                    qchg = (comp.loc["Post", "Mean daily units"] / comp.loc["Pre", "Mean daily units"] - 1) * 100
                    u, v = st.columns(2)
                    u.metric("Mean price change", f"{pchg:+.2f}%")
                    v.metric("Mean daily units change", f"{qchg:+.2f}%")
            else:
                st.info("The selected date window needs Bengaluru observations on both sides of the split date.")
        st.caption("Revenue proxy = Σ(price × units). It is not audited revenue unless price and sales represent realized transaction values. Pre/post differences are descriptive.")

elif page == "Data Explorer":
    st.header("Data Explorer")
    tab1, tab2, tab3 = st.tabs(["Regional data", "Synthetic control", "Data quality audit"])
    with tab1:
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button("Download filtered regional CSV", filtered.to_csv(index=False).encode("utf-8"), "regional_sales_filtered.csv", "text/csv")
    with tab2:
        st.dataframe(sc_view, use_container_width=True, hide_index=True)
        st.download_button("Download filtered synthetic CSV", sc_view.to_csv(index=False).encode("utf-8"), "synthetic_control_filtered.csv", "text/csv")
    with tab3:
        audit_view = {k: (v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else v) for k, v in audit.items()}
        st.json(audit_view)
        if audit["bengaluru_reconciliation_mismatches"]:
            st.error("Stored Bengaluru actuals do not fully reconcile with the regional dataset.")
        else:
            st.success("Stored Bengaluru actuals reconcile with the regional dataset on overlapping dates.")
        st.caption("Reconciliation checks the actual Bengaluru series in both supplied CSVs; it does not validate the synthetic counterfactual methodology.")

st.divider()
st.caption("Algorithmic Pricing Analysis · Streamlit + Pandas + Plotly · Simulated data · Reproducible metrics")
