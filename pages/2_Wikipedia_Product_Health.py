"""Wikipedia Product Health Analysis — Streamlit Dashboard (demo data)."""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="Wikipedia Product Health",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLATFORMS = ["desktop", "mobile-web", "mobile-app"]
METRICS = ["pageviews", "editors", "edits"]
N_DAYS = 730  # 2 years of data


@st.cache_data
def gen_timeseries(n_days: int = N_DAYS) -> pd.DataFrame:
    random.seed(21)
    base = datetime(2022, 1, 1)
    rows = []
    for d in range(n_days):
        dt = base + timedelta(days=d)
        week = d // 7
        # seasonal: weekly + annual cycle
        weekly_season = 1 + 0.15 * np.sin(2 * np.pi * d / 7)
        annual_season = 1 + 0.20 * np.sin(2 * np.pi * d / 365)
        # structural shift at day 400
        shift = 1.12 if d > 400 else 1.0
        for platform in PLATFORMS:
            base_pv = {"desktop": 4_000_000, "mobile-web": 3_200_000, "mobile-app": 1_800_000}[platform]
            base_ed = {"desktop": 12_000, "mobile-web": 4_000, "mobile-app": 1_500}[platform]
            base_edits = {"desktop": 80_000, "mobile-web": 20_000, "mobile-app": 8_000}[platform]
            trend = 1 + d * 0.0002
            noise = random.gauss(1.0, 0.04)
            rows.append({
                "date": dt,
                "platform": platform,
                "pageviews": int(base_pv * trend * weekly_season * annual_season * shift * noise),
                "editors": int(base_ed * trend * annual_season * shift * random.gauss(1.0, 0.06)),
                "edits": int(base_edits * trend * weekly_season * shift * random.gauss(1.0, 0.05)),
            })
    return pd.DataFrame(rows)


@st.cache_data
def gen_platform_risk() -> pd.DataFrame:
    random.seed(77)
    rows = []
    for platform in PLATFORMS:
        share = {"desktop": 0.44, "mobile-web": 0.35, "mobile-app": 0.21}[platform]
        cagr = {"desktop": -2.1, "mobile-web": 8.4, "mobile-app": 14.2}[platform]
        vol = {"desktop": 0.08, "mobile-web": 0.12, "mobile-app": 0.18}[platform]
        rows.append({
            "platform": platform, "share": share, "cagr": cagr,
            "volatility": vol,
            "hhi_contribution": round(share ** 2 * 10000, 1),
        })
    df = pd.DataFrame(rows)
    df["hhi"] = df["hhi_contribution"].sum()
    return df


@st.cache_data
def gen_changepoints() -> pd.DataFrame:
    random.seed(33)
    cps = [
        {"date": datetime(2022, 6, 15), "platform": "mobile-app", "magnitude": 0.18,
         "direction": "increase", "confidence": 0.94, "cause": "App redesign launch"},
        {"date": datetime(2022, 11, 3), "platform": "desktop", "magnitude": -0.09,
         "direction": "decrease", "confidence": 0.87, "cause": "Algorithm change"},
        {"date": datetime(2023, 3, 22), "platform": "mobile-web", "magnitude": 0.14,
         "direction": "increase", "confidence": 0.91, "cause": "SEO improvements"},
        {"date": datetime(2023, 8, 10), "platform": "all", "magnitude": 0.12,
         "direction": "increase", "confidence": 0.96, "cause": "Wikipedia 22nd anniversary campaign"},
        {"date": datetime(2024, 1, 5), "platform": "desktop", "magnitude": -0.07,
         "direction": "decrease", "confidence": 0.82, "cause": "Holiday season dip"},
    ]
    return pd.DataFrame(cps)


@st.cache_data
def gen_causal_effects() -> pd.DataFrame:
    random.seed(44)
    effects = [
        {"event": "App Redesign (Jun 2022)", "platform": "mobile-app",
         "effect_size": 0.18, "ci_lower": 0.12, "ci_upper": 0.24,
         "p_value": 0.001, "method": "ITS", "significant": True},
        {"event": "SEO Campaign (Mar 2023)", "platform": "mobile-web",
         "effect_size": 0.14, "ci_lower": 0.08, "ci_upper": 0.20,
         "p_value": 0.003, "method": "DiD", "significant": True},
        {"event": "Algorithm Change (Nov 2022)", "platform": "desktop",
         "effect_size": -0.09, "ci_lower": -0.15, "ci_upper": -0.03,
         "p_value": 0.012, "method": "ITS", "significant": True},
        {"event": "Anniversary Campaign (Aug 2023)", "platform": "all",
         "effect_size": 0.12, "ci_lower": 0.07, "ci_upper": 0.17,
         "p_value": 0.002, "method": "Event Study", "significant": True},
        {"event": "Holiday Dip (Jan 2024)", "platform": "desktop",
         "effect_size": -0.07, "ci_lower": -0.14, "ci_upper": 0.00,
         "p_value": 0.051, "method": "ITS", "significant": False},
    ]
    return pd.DataFrame(effects)


@st.cache_data
def gen_forecast_health(platform: str, days_ahead: int = 90) -> pd.DataFrame:
    random.seed(hash(platform) % 500)
    base = datetime(2024, 1, 1)
    base_val = {"desktop": 4_500_000, "mobile-web": 3_800_000, "mobile-app": 2_200_000}[platform]
    rows = []
    for d in range(days_ahead):
        dt = base + timedelta(days=d)
        trend = base_val * (1 + d * 0.0003)
        noise = random.gauss(0, base_val * 0.03)
        yhat = max(0, trend + noise)
        rows.append({"date": dt, "yhat": yhat,
                     "yhat_lower": yhat * 0.88, "yhat_upper": yhat * 1.12})
    return pd.DataFrame(rows)


@st.cache_data
def gen_seasonality() -> pd.DataFrame:
    random.seed(66)
    rows = []
    for week in range(52):
        rows.append({
            "week": week + 1,
            "avg_pageviews": 9_000_000 + 800_000 * np.sin(2 * np.pi * week / 52) + random.gauss(0, 100_000),
            "avg_editors": 17_000 + 2_000 * np.sin(2 * np.pi * week / 52) + random.gauss(0, 200),
        })
    return pd.DataFrame(rows)


# -- sidebar -------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/health-checkup.png", width=60)
st.sidebar.title("Wikipedia Product Health")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Traffic Trends", "Platform Risk", "Changepoint Detection",
     "Causal Analysis", "Seasonality", "Forecasting"],
)

platform_filter = st.sidebar.multiselect("Platforms", PLATFORMS, default=PLATFORMS)
date_range = st.sidebar.selectbox("Date Range", ["Last 90 days", "Last 180 days",
                                                  "Last 365 days", "All (2 years)"], index=2)
days_map = {"Last 90 days": 90, "Last 180 days": 180, "Last 365 days": 365, "All (2 years)": N_DAYS}
n_days_show = days_map[date_range]

ts_df = gen_timeseries()
risk_df = gen_platform_risk()
cp_df = gen_changepoints()
causal_df = gen_causal_effects()
season_df = gen_seasonality()

cutoff = ts_df["date"].max() - timedelta(days=n_days_show)
ts_filtered = ts_df[(ts_df["date"] >= cutoff) & (ts_df["platform"].isin(platform_filter))]


def kpi(col, label, value, delta=None, fmt="{:,.0f}"):
    col.metric(label, fmt.format(value) if isinstance(value, (int, float)) else value,
               delta=delta)


# -- OVERVIEW ------------------------------------------------------------------
if page == "Overview":
    st.title("🏥 Wikipedia Product Health Analysis")
    st.caption(f"Demo dataset · {date_range} · {len(platform_filter)} platform(s)")
    st.markdown("---")

    total_pv = ts_filtered["pageviews"].sum()
    total_ed = ts_filtered["editors"].sum()
    total_edits = ts_filtered["edits"].sum()
    hhi = risk_df["hhi"].iloc[0]
    mobile_share = risk_df[risk_df["platform"].isin(["mobile-web", "mobile-app"])]["share"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "Total Pageviews", total_pv)
    kpi(c2, "Total Editors", total_ed)
    kpi(c3, "Total Edits", total_edits)
    c4.metric("HHI (Platform Conc.)", f"{hhi:.0f}", delta=None,
              help="Herfindahl-Hirschman Index — >2500 = high concentration")
    c5.metric("Mobile Share", f"{mobile_share:.0%}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Daily Pageviews by Platform")
        daily_pv = ts_filtered.groupby(["date", "platform"])["pageviews"].sum().reset_index()
        fig = px.area(daily_pv, x="date", y="pageviews", color="platform",
                      labels={"pageviews": "Pageviews", "date": ""},
                      template="plotly_white",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=350, legend_title_text="Platform")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Platform Share")
        fig = px.pie(risk_df, names="platform", values="share",
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     template="plotly_white")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Editors & Edits Over Time")
    daily_ee = ts_filtered.groupby("date")[["editors", "edits"]].sum().reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=daily_ee["date"], y=daily_ee["editors"],
                             name="Editors", line_color="#636EFA"), secondary_y=False)
    fig.add_trace(go.Scatter(x=daily_ee["date"], y=daily_ee["edits"],
                             name="Edits", line_color="#EF553B"), secondary_y=True)
    fig.update_layout(template="plotly_white", height=280,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(title_text="Editors", secondary_y=False)
    fig.update_yaxes(title_text="Edits", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)


# -- TRAFFIC TRENDS ------------------------------------------------------------
elif page == "Traffic Trends":
    st.title("📉 Traffic Trends")
    st.caption("Long-term trend decomposition and growth analysis")
    st.markdown("---")

    metric = st.selectbox("Metric", METRICS)

    daily = ts_filtered.groupby(["date", "platform"])[metric].sum().reset_index()
    daily_total = ts_filtered.groupby("date")[metric].sum().reset_index()

    # rolling 7-day average
    daily_total["rolling_7d"] = daily_total[metric].rolling(7, min_periods=1).mean()
    daily_total["rolling_30d"] = daily_total[metric].rolling(30, min_periods=1).mean()

    c1, c2, c3 = st.columns(3)
    first_val = daily_total[metric].iloc[:30].mean()
    last_val = daily_total[metric].iloc[-30:].mean()
    growth = (last_val - first_val) / first_val * 100
    c1.metric(f"Avg {metric.title()} (first 30d)", f"{first_val:,.0f}")
    c2.metric(f"Avg {metric.title()} (last 30d)", f"{last_val:,.0f}")
    c3.metric("Period Growth", f"{growth:+.1f}%")

    st.markdown("---")
    st.subheader(f"{metric.title()} — Raw + Rolling Averages")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_total["date"], y=daily_total[metric],
                             name="Daily", opacity=0.3, line=dict(color="#aaa", width=1)))
    fig.add_trace(go.Scatter(x=daily_total["date"], y=daily_total["rolling_7d"],
                             name="7-day MA", line=dict(color="#636EFA", width=2)))
    fig.add_trace(go.Scatter(x=daily_total["date"], y=daily_total["rolling_30d"],
                             name="30-day MA", line=dict(color="#EF553B", width=2.5)))
    fig.update_layout(template="plotly_white", height=380,
                      xaxis_title="", yaxis_title=metric.title(),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader(f"{metric.title()} by Platform")
        fig = px.line(daily, x="date", y=metric, color="platform",
                      labels={metric: metric.title(), "date": ""},
                      template="plotly_white",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=320, legend_title_text="Platform")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Month-over-Month Growth (%)")
        monthly = daily_total.copy()
        monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
        monthly_agg = monthly.groupby("month")[metric].mean().reset_index()
        monthly_agg["mom_growth"] = monthly_agg[metric].pct_change() * 100
        fig = px.bar(monthly_agg.dropna(), x="month", y="mom_growth",
                     color="mom_growth", color_continuous_scale="RdYlGn",
                     labels={"mom_growth": "MoM Growth (%)", "month": ""},
                     template="plotly_white")
        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.update_layout(coloraxis_showscale=False, height=320, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


# -- PLATFORM RISK -------------------------------------------------------------
elif page == "Platform Risk":
    st.title("⚠️ Platform Risk Assessment")
    st.caption("HHI concentration, CAGR, volatility, and mobile dependency analysis")
    st.markdown("---")

    hhi = risk_df["hhi"].iloc[0]
    mobile_share = risk_df[risk_df["platform"].isin(["mobile-web", "mobile-app"])]["share"].sum()
    risk_level = "High" if hhi > 2500 else ("Moderate" if hhi > 1500 else "Low")
    mobile_risk = "High" if mobile_share > 0.70 else ("Moderate" if mobile_share > 0.55 else "Low")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HHI Score", f"{hhi:.0f}", help=">2500 = high concentration")
    c2.metric("Concentration Risk", risk_level)
    c3.metric("Mobile Share", f"{mobile_share:.0%}")
    c4.metric("Mobile Dependency Risk", mobile_risk)

    if hhi > 2500:
        st.error("🔴 High platform concentration detected (HHI > 2500). Diversification recommended.")
    elif hhi > 1500:
        st.warning("🟡 Moderate platform concentration (HHI 1500–2500).")
    else:
        st.success("🟢 Low platform concentration (HHI < 1500).")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Platform Share Distribution")
        fig = px.pie(risk_df, names="platform", values="share",
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     template="plotly_white")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("CAGR by Platform")
        fig = px.bar(risk_df, x="platform", y="cagr",
                     color="cagr", color_continuous_scale="RdYlGn",
                     labels={"cagr": "CAGR (%)", "platform": ""},
                     template="plotly_white")
        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Volatility & HHI Contribution")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=risk_df["platform"], y=risk_df["volatility"],
                         name="Volatility (CoV)", marker_color="#636EFA"))
    fig.add_trace(go.Bar(x=risk_df["platform"],
                         y=risk_df["hhi_contribution"] / 10000,
                         name="HHI Contribution (norm.)", marker_color="#EF553B"))
    fig.update_layout(barmode="group", template="plotly_white", height=320,
                      yaxis_title="Value", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Scenario Analysis — Mobile Decline Impact")
    scenarios = []
    for decline in [0.05, 0.10, 0.20, 0.30]:
        mobile_pv = ts_filtered[ts_filtered["platform"].isin(["mobile-web", "mobile-app"])]["pageviews"].sum()
        total_pv = ts_filtered["pageviews"].sum()
        impact = mobile_pv * decline
        scenarios.append({
            "Mobile Decline": f"{decline:.0%}",
            "Views Lost": f"{impact:,.0f}",
            "Total Impact %": f"{impact / total_pv * 100:.1f}%",
        })
    st.table(pd.DataFrame(scenarios))


# -- CHANGEPOINT DETECTION -----------------------------------------------------
elif page == "Changepoint Detection":
    st.title("🔍 Changepoint Detection")
    st.caption("Structural breaks detected via PELT algorithm with Chow test significance")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("Changepoints Detected", len(cp_df))
    c2.metric("Avg Confidence", f"{cp_df['confidence'].mean():.2f}")
    c3.metric("Significant Shifts", len(cp_df[cp_df["confidence"] > 0.90]))

    st.markdown("---")
    # Show pageviews with changepoint annotations
    daily_total = ts_filtered.groupby("date")["pageviews"].sum().reset_index()
    daily_total["rolling_7d"] = daily_total["pageviews"].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_total["date"], y=daily_total["rolling_7d"],
                             name="7-day MA Pageviews", line=dict(color="#636EFA", width=2)))
    colors = {"increase": "#00CC96", "decrease": "#EF553B"}
    for _, cp in cp_df.iterrows():
        if cp["date"] >= daily_total["date"].min():
            fig.add_vline(
                x=cp["date"].timestamp() * 1000,
                line_dash="dash",
                line_color=colors.get(cp["direction"], "gray"),
                annotation_text=f"{cp['cause'][:20]}…" if len(cp["cause"]) > 20 else cp["cause"],
                annotation_position="top",
                annotation_font_size=10,
            )
    fig.update_layout(template="plotly_white", height=420,
                      xaxis_title="", yaxis_title="Pageviews (7d MA)",
                      title="Pageview Trend with Detected Changepoints")
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Changepoint Magnitude")
        fig = px.bar(cp_df.sort_values("magnitude"),
                     x="magnitude", y="cause", orientation="h",
                     color="direction",
                     color_discrete_map={"increase": "#00CC96", "decrease": "#EF553B"},
                     labels={"magnitude": "Magnitude", "cause": ""},
                     template="plotly_white")
        fig.add_vline(x=0, line_color="black", line_width=1)
        fig.update_layout(height=320, legend_title_text="Direction")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Confidence by Changepoint")
        fig = px.bar(cp_df.sort_values("confidence", ascending=False),
                     x="cause", y="confidence",
                     color="confidence", color_continuous_scale="RdYlGn",
                     labels={"confidence": "Confidence", "cause": ""},
                     template="plotly_white")
        fig.add_hline(y=0.90, line_dash="dash", annotation_text="0.90 threshold")
        fig.update_layout(coloraxis_showscale=False, height=320, xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Changepoint Details")
    st.dataframe(cp_df[["date", "platform", "cause", "direction", "magnitude", "confidence"]]
                 .sort_values("date"),
                 use_container_width=True)


# -- CAUSAL ANALYSIS -----------------------------------------------------------
elif page == "Causal Analysis":
    st.title("🔬 Causal Analysis")
    st.caption("Interrupted Time Series (ITS), Difference-in-Differences (DiD), and Event Study results")
    st.markdown("---")

    sig_effects = causal_df[causal_df["significant"]]
    c1, c2, c3 = st.columns(3)
    c1.metric("Events Analyzed", len(causal_df))
    c2.metric("Significant Effects", len(sig_effects))
    c3.metric("Avg Effect Size", f"{causal_df['effect_size'].mean():+.3f}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Effect Sizes with 95% CI")
        fig = go.Figure()
        for i, row in causal_df.iterrows():
            color = "#00CC96" if row["significant"] else "#aaa"
            fig.add_trace(go.Scatter(
                x=[row["ci_lower"], row["ci_upper"]],
                y=[row["event"], row["event"]],
                mode="lines", line=dict(color=color, width=3),
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[row["effect_size"]], y=[row["event"]],
                mode="markers",
                marker=dict(size=12, color=color,
                            symbol="diamond" if row["significant"] else "circle"),
                name=row["event"], showlegend=False,
            ))
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        fig.update_layout(template="plotly_white", height=380,
                          xaxis_title="Effect Size", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("P-values by Event")
        fig = px.bar(causal_df.sort_values("p_value"),
                     x="p_value", y="event", orientation="h",
                     color="significant",
                     color_discrete_map={True: "#00CC96", False: "#EF553B"},
                     labels={"p_value": "p-value", "event": ""},
                     template="plotly_white")
        fig.add_vline(x=0.05, line_dash="dash", annotation_text="α=0.05")
        fig.update_layout(height=380, legend_title_text="Significant")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Before/After Comparison — App Redesign (Jun 2022)")
    before_end = datetime(2022, 6, 14)
    after_start = datetime(2022, 6, 16)
    before = ts_df[(ts_df["date"] <= before_end) & (ts_df["platform"] == "mobile-app")]
    after = ts_df[(ts_df["date"] >= after_start) & (ts_df["platform"] == "mobile-app")]
    before_avg = before["pageviews"].mean()
    after_avg = after["pageviews"].mean()
    lift = (after_avg - before_avg) / before_avg * 100

    ba1, ba2, ba3 = st.columns(3)
    ba1.metric("Before Avg Pageviews", f"{before_avg:,.0f}")
    ba2.metric("After Avg Pageviews", f"{after_avg:,.0f}")
    ba3.metric("Lift", f"{lift:+.1f}%")

    st.subheader("Full Causal Effects Table")
    st.dataframe(
        causal_df[["event", "platform", "method", "effect_size",
                   "ci_lower", "ci_upper", "p_value", "significant"]],
        use_container_width=True,
    )


# -- SEASONALITY ---------------------------------------------------------------
elif page == "Seasonality":
    st.title("📅 Seasonality Analysis")
    st.caption("Weekly and annual seasonal patterns in pageviews and editor activity")
    st.markdown("---")

    # Day-of-week pattern
    ts_filtered_copy = ts_filtered.copy()
    ts_filtered_copy["dow"] = ts_filtered_copy["date"].dt.day_name()
    ts_filtered_copy["month"] = ts_filtered_copy["date"].dt.month_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    dow_agg = ts_filtered_copy.groupby("dow")["pageviews"].mean().reindex(dow_order).reset_index()
    month_agg = ts_filtered_copy.groupby("month")["pageviews"].mean().reindex(month_order).reset_index()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Avg Pageviews by Day of Week")
        fig = px.bar(dow_agg, x="dow", y="pageviews",
                     color="pageviews", color_continuous_scale="Blues",
                     labels={"pageviews": "Avg Pageviews", "dow": ""},
                     template="plotly_white")
        fig.update_layout(coloraxis_showscale=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Avg Pageviews by Month")
        fig = px.bar(month_agg, x="month", y="pageviews",
                     color="pageviews", color_continuous_scale="Oranges",
                     labels={"pageviews": "Avg Pageviews", "month": ""},
                     template="plotly_white")
        fig.update_layout(coloraxis_showscale=False, height=320, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Weekly Seasonal Pattern (52-week cycle)")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=season_df["week"], y=season_df["avg_pageviews"],
                             name="Avg Pageviews", line_color="#636EFA"), secondary_y=False)
    fig.add_trace(go.Scatter(x=season_df["week"], y=season_df["avg_editors"],
                             name="Avg Editors", line_color="#EF553B"), secondary_y=True)
    fig.update_layout(template="plotly_white", height=320,
                      xaxis_title="Week of Year",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(title_text="Pageviews", secondary_y=False)
    fig.update_yaxes(title_text="Editors", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Heatmap — Pageviews by Day of Week × Month")
    ts_filtered_copy["month_num"] = ts_filtered_copy["date"].dt.month
    heat = ts_filtered_copy.groupby(["dow", "month"])["pageviews"].mean().reset_index()
    heat_pivot = heat.pivot(index="dow", columns="month", values="pageviews")
    heat_pivot = heat_pivot.reindex(dow_order)
    heat_pivot = heat_pivot.reindex(columns=month_order)
    fig = px.imshow(heat_pivot, color_continuous_scale="Blues",
                    labels={"color": "Avg Pageviews"},
                    aspect="auto", template="plotly_white")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


# -- FORECASTING ---------------------------------------------------------------
elif page == "Forecasting":
    st.title("📈 Forecasting")
    st.caption("90-day ensemble forecast with prediction intervals per platform")
    st.markdown("---")

    forecast_platform = st.selectbox("Platform", PLATFORMS)
    horizon = st.slider("Forecast Horizon (days)", 30, 180, 90)
    fc_df = gen_forecast_health(forecast_platform, days_ahead=horizon)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast Start", fc_df["date"].min().strftime("%Y-%m-%d"))
    c2.metric("Forecast End", fc_df["date"].max().strftime("%Y-%m-%d"))
    c3.metric("Avg Forecast", f"{fc_df['yhat'].mean():,.0f}")
    growth = (fc_df["yhat"].iloc[-1] - fc_df["yhat"].iloc[0]) / fc_df["yhat"].iloc[0] * 100
    c4.metric("Forecast Growth", f"{growth:+.1f}%")

    st.markdown("---")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fc_df["date"], y=fc_df["yhat_upper"],
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=fc_df["date"], y=fc_df["yhat_lower"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(99,110,250,0.15)",
        name="95% Prediction Interval",
    ))
    fig.add_trace(go.Scatter(
        x=fc_df["date"], y=fc_df["yhat"],
        mode="lines", name="Forecast",
        line=dict(color="#636EFA", width=2.5),
    ))
    fig.update_layout(
        title=f"Pageview Forecast — {forecast_platform}",
        xaxis_title="", yaxis_title="Predicted Pageviews",
        template="plotly_white", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("All Platforms — Forecast Comparison")
    all_fc = []
    for p in PLATFORMS:
        fc = gen_forecast_health(p, days_ahead=90)
        fc["platform"] = p
        all_fc.append(fc)
    all_fc_df = pd.concat(all_fc)

    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.line(all_fc_df, x="date", y="yhat", color="platform",
                      labels={"yhat": "Forecast Pageviews", "date": ""},
                      template="plotly_white",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=320, legend_title_text="Platform")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        summary = []
        for p in PLATFORMS:
            fc = gen_forecast_health(p, days_ahead=90)
            g = (fc["yhat"].iloc[-1] - fc["yhat"].iloc[0]) / fc["yhat"].iloc[0] * 100
            summary.append({
                "platform": p,
                "avg_forecast": round(fc["yhat"].mean(), 0),
                "growth_pct": round(g, 1),
            })
        summary_df = pd.DataFrame(summary)
        fig = px.bar(summary_df, x="platform", y="growth_pct",
                     color="growth_pct", color_continuous_scale="RdYlGn",
                     labels={"growth_pct": "Forecast Growth (%)", "platform": ""},
                     template="plotly_white")
        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.update_layout(coloraxis_showscale=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Accuracy Metrics (simulated)")
    acc_data = {
        "Platform": PLATFORMS,
        "MAE": [random.randint(80_000, 200_000) for _ in PLATFORMS],
        "RMSE": [random.randint(100_000, 250_000) for _ in PLATFORMS],
        "MAPE (%)": [round(random.uniform(2.5, 8.0), 2) for _ in PLATFORMS],
        "Method": ["Ensemble (ARIMA + Prophet + ETS)"] * 3,
    }
    st.dataframe(pd.DataFrame(acc_data), use_container_width=True)
