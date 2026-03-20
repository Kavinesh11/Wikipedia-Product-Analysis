"""Fortune 500 Knowledge Graph Analytics — Streamlit Dashboard.

Run with:
    streamlit run fortune500_kg/dashboard_app.py
"""

import random
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fortune 500 KG Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── demo data generator ───────────────────────────────────────────────────────
SECTORS = ["Technology", "Healthcare", "Finance", "Energy", "Retail",
           "Manufacturing", "Telecom", "Consumer Goods"]

@st.cache_data
def generate_demo_data(n: int = 100) -> pd.DataFrame:
    random.seed(42)
    companies = []
    for i in range(1, n + 1):
        sector = random.choice(SECTORS)
        stars = random.randint(100, 50_000)
        forks = random.randint(10, 15_000)
        employees = random.randint(500, 200_000)
        contributors = random.randint(5, 2_000)
        revenue_rank = i
        raw_score = (stars + forks) / employees
        dmi = (stars + forks + contributors) / revenue_rank
        companies.append({
            "company_id": f"C{i:03d}",
            "name": f"Company {i:03d}",
            "sector": sector,
            "revenue_rank": revenue_rank,
            "employee_count": employees,
            "github_stars": stars,
            "github_forks": forks,
            "contributors": contributors,
            "raw_innovation_score": raw_score,
            "digital_maturity_index": dmi,
            "pagerank": random.uniform(0.001, 0.05),
            "ecosystem_centrality": random.uniform(0.0, 0.3),
            "community_id": random.randint(0, 7),
            "revenue_growth": random.uniform(-5.0, 25.0),
            "predicted_growth": random.uniform(-3.0, 22.0),
            "confidence_score": random.uniform(0.4, 0.99),
            "ma_activity": random.randint(0, 10),
        })
    df = pd.DataFrame(companies)
    # normalise innovation score 0-10
    mn, mx = df["raw_innovation_score"].min(), df["raw_innovation_score"].max()
    df["innovation_score"] = (df["raw_innovation_score"] - mn) / (mx - mn) * 10
    df["decile"] = pd.qcut(df["innovation_score"], 10, labels=False) + 1
    df["high_confidence"] = df["confidence_score"] > 0.80
    return df


@st.cache_data
def generate_trend_data(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = datetime(2021, 1, 1)
    for month in range(36):
        dt = base + timedelta(days=month * 30)
        noise = random.gauss(0, 0.3)
        rows.append({
            "date": dt,
            "avg_innovation_score": 4.5 + month * 0.05 + noise,
            "avg_digital_maturity": 120 + month * 2.1 + random.gauss(0, 5),
        })
    return pd.DataFrame(rows)


@st.cache_data
def generate_sector_trend(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = datetime(2021, 1, 1)
    for sector in SECTORS:
        base_score = random.uniform(3.0, 7.0)
        for month in range(36):
            dt = base + timedelta(days=month * 30)
            rows.append({
                "date": dt,
                "sector": sector,
                "avg_innovation_score": base_score + month * 0.04 + random.gauss(0, 0.2),
            })
    return pd.DataFrame(rows)


# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/wikipedia.png", width=60)
st.sidebar.title("Fortune 500 KG Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Leaderboard", "Sector Analysis", "Network & Clusters",
     "Predictions", "ROI & Insights", "Custom Query"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
df_all = generate_demo_data(150)
selected_sectors = st.sidebar.multiselect(
    "Sectors", SECTORS, default=SECTORS
)
top_n = st.sidebar.slider("Top N companies", 10, 150, 50)
year_filter = st.sidebar.selectbox("Year", [2021, 2022, 2023, 2024], index=3)

df = df_all[df_all["sector"].isin(selected_sectors)].head(top_n).copy()
trend_df = generate_trend_data(df)
sector_trend_df = generate_sector_trend(df)


# ── helpers ───────────────────────────────────────────────────────────────────
def kpi(col, label, value, delta=None, fmt="{:.2f}"):
    col.metric(label, fmt.format(value) if isinstance(value, float) else value,
               delta=f"{delta:+.2f}" if delta is not None else None)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title(" Fortune 500 Knowledge Graph Analytics")
    st.caption(f"Demo dataset · {len(df)} companies · {len(selected_sectors)} sectors")
    st.markdown("---")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "Companies", len(df), fmt="{}")
    kpi(c2, "Avg Innovation Score", df["innovation_score"].mean(), delta=0.3)
    kpi(c3, "Avg Digital Maturity", df["digital_maturity_index"].mean(), delta=12.1)
    kpi(c4, "High-Confidence Predictions",
        int(df["high_confidence"].sum()), fmt="{}")
    kpi(c5, "Sectors Covered", len(df["sector"].unique()), fmt="{}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Innovation Score Distribution")
        fig = px.histogram(df, x="innovation_score", nbins=20,
                           color="sector", barmode="overlay",
                           labels={"innovation_score": "Innovation Score (0-10)"},
                           template="plotly_white")
        fig.update_layout(legend_title_text="Sector", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Digital Maturity by Sector")
        sector_dmi = df.groupby("sector")["digital_maturity_index"].mean().reset_index()
        fig = px.bar(sector_dmi.sort_values("digital_maturity_index", ascending=True),
                     x="digital_maturity_index", y="sector", orientation="h",
                     color="digital_maturity_index",
                     color_continuous_scale="Blues",
                     labels={"digital_maturity_index": "Avg DMI", "sector": ""},
                     template="plotly_white")
        fig.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Innovation Score Trend (2021–2024)")
    fig = px.line(trend_df, x="date", y="avg_innovation_score",
                  labels={"avg_innovation_score": "Avg Innovation Score", "date": ""},
                  template="plotly_white")
    fig.update_traces(line_color="#636EFA", line_width=2.5)
    fig.update_layout(height=280)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Leaderboard":
    st.title("Innovation Score Leaderboard")
    st.caption("Ranked by normalised Innovation Score (0–10)")
    st.markdown("---")

    top_df = df.sort_values("innovation_score", ascending=False).reset_index(drop=True)
    top_df.index += 1
    top_df["rank"] = top_df.index

    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.subheader("Top Companies — Bar Chart")
        show_n = st.slider("Show top N", 10, min(50, len(top_df)), 20, key="lb_n")
        fig = px.bar(top_df.head(show_n),
                     x="innovation_score", y="name",
                     orientation="h",
                     color="sector",
                     hover_data=["revenue_rank", "employee_count",
                                 "github_stars", "github_forks"],
                     labels={"innovation_score": "Innovation Score", "name": ""},
                     template="plotly_white")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Decile Distribution")
        decile_counts = top_df["decile"].value_counts().sort_index().reset_index()
        decile_counts.columns = ["Decile", "Count"]
        fig = px.bar(decile_counts, x="Decile", y="Count",
                     color="Count", color_continuous_scale="Viridis",
                     template="plotly_white")
        fig.update_layout(coloraxis_showscale=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Innovation vs Revenue Rank")
        fig = px.scatter(top_df, x="revenue_rank", y="innovation_score",
                         color="sector", size="employee_count",
                         hover_name="name",
                         labels={"revenue_rank": "Fortune 500 Rank",
                                 "innovation_score": "Innovation Score"},
                         template="plotly_white")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Full Leaderboard Table")
    display_cols = ["rank", "name", "sector", "innovation_score",
                    "digital_maturity_index", "ecosystem_centrality",
                    "revenue_rank", "github_stars"]
    st.dataframe(
        top_df[display_cols].rename(columns={
            "innovation_score": "Innovation Score",
            "digital_maturity_index": "DMI",
            "ecosystem_centrality": "Centrality",
            "revenue_rank": "F500 Rank",
            "github_stars": "Stars",
        }),
        use_container_width=True, height=400,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SECTOR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Sector Analysis":
    st.title(" Cross-Sector Comparative Analysis")
    st.markdown("---")

    sector_agg = df.groupby("sector").agg(
        avg_innovation=("innovation_score", "mean"),
        avg_dmi=("digital_maturity_index", "mean"),
        avg_centrality=("ecosystem_centrality", "mean"),
        avg_pagerank=("pagerank", "mean"),
        company_count=("company_id", "count"),
    ).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Avg Innovation Score by Sector")
        fig = px.bar(sector_agg.sort_values("avg_innovation", ascending=False),
                     x="sector", y="avg_innovation",
                     color="avg_innovation", color_continuous_scale="RdYlGn",
                     labels={"avg_innovation": "Avg Innovation Score", "sector": ""},
                     template="plotly_white")
        fig.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Sector Radar — Key Metrics")
        radar_df = sector_agg.copy()
        for col in ["avg_innovation", "avg_dmi", "avg_centrality", "avg_pagerank"]:
            mn, mx = radar_df[col].min(), radar_df[col].max()
            radar_df[col] = (radar_df[col] - mn) / (mx - mn + 1e-9)
        categories = ["Innovation", "Digital Maturity", "Centrality", "PageRank"]
        fig = go.Figure()
        for _, row in radar_df.iterrows():
            vals = [row["avg_innovation"], row["avg_dmi"],
                    row["avg_centrality"], row["avg_pagerank"]]
            vals += [vals[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=categories + [categories[0]],
                fill="toself", name=row["sector"], opacity=0.6,
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                          height=380, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sector Trend — Innovation Score Over Time")
    fig = px.line(sector_trend_df[sector_trend_df["sector"].isin(selected_sectors)],
                  x="date", y="avg_innovation_score", color="sector",
                  labels={"avg_innovation_score": "Avg Innovation Score", "date": ""},
                  template="plotly_white")
    fig.update_layout(height=320, legend_title_text="Sector")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Heatmap — Innovation Score × Sector × Decile")
    heat_df = df.groupby(["sector", "decile"])["innovation_score"].mean().reset_index()
    heat_pivot = heat_df.pivot(index="sector", columns="decile", values="innovation_score")
    fig = px.imshow(heat_pivot, color_continuous_scale="Blues",
                    labels={"color": "Avg Innovation Score"},
                    aspect="auto", template="plotly_white")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Inter-Sector Percentage Differences (Innovation Score)")
    best = sector_agg.loc[sector_agg["avg_innovation"].idxmax(), "sector"]
    worst = sector_agg.loc[sector_agg["avg_innovation"].idxmin(), "sector"]
    best_val = sector_agg.loc[sector_agg["sector"] == best, "avg_innovation"].values[0]
    worst_val = sector_agg.loc[sector_agg["sector"] == worst, "avg_innovation"].values[0]
    pct_diff = (best_val - worst_val) / worst_val * 100
    st.info(f"**{best}** leads **{worst}** by **{pct_diff:.1f}%** in average Innovation Score.")
    st.dataframe(sector_agg.rename(columns={
        "avg_innovation": "Avg Innovation", "avg_dmi": "Avg DMI",
        "avg_centrality": "Avg Centrality", "avg_pagerank": "Avg PageRank",
        "company_count": "# Companies",
    }),
        use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: NETWORK & CLUSTERS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Network & Clusters":
    st.title(" Network Graph & Cluster Detection")
    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Community Clusters (Louvain)")
        cluster_counts = df.groupby("community_id").agg(
            companies=("company_id", "count"),
            avg_innovation=("innovation_score", "mean"),
            avg_centrality=("ecosystem_centrality", "mean"),
        ).reset_index()
        cluster_counts["density"] = (
            cluster_counts["companies"] / cluster_counts["companies"].sum()
        )
        fig = px.scatter(cluster_counts,
                         x="avg_innovation", y="avg_centrality",
                         size="companies", color="community_id",
                         hover_data=["companies", "density"],
                         labels={"avg_innovation": "Avg Innovation Score",
                                 "avg_centrality": "Avg Centrality",
                                 "community_id": "Cluster"},
                         template="plotly_white",
                         color_continuous_scale="Turbo")
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Network Density per Cluster")
        fig = px.bar(cluster_counts.sort_values("density", ascending=False),
                     x="community_id", y="density",
                     color="density", color_continuous_scale="RdYlGn",
                     labels={"community_id": "Cluster ID", "density": "Density"},
                     template="plotly_white")
        median_density = cluster_counts["density"].median()
        fig.add_hline(y=median_density, line_dash="dash", line_color="red",
                      annotation_text=f"Median: {median_density:.3f}")
        fig.update_layout(coloraxis_showscale=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Force-Directed Network (Sampled — top 60 by centrality)")
    sample = df.nlargest(60, "ecosystem_centrality").reset_index(drop=True)
    # Build edges: connect companies in same community
    edge_x, edge_y = [], []
    import math
    n = len(sample)
    angles = [2 * math.pi * i / n for i in range(n)]
    xs = [math.cos(a) * (1 + sample.loc[i, "ecosystem_centrality"] * 2) for i, a in enumerate(angles)]
    ys = [math.sin(a) * (1 + sample.loc[i, "ecosystem_centrality"] * 2) for i, a in enumerate(angles)]
    for i in range(n):
        for j in range(i + 1, n):
            if sample.loc[i, "community_id"] == sample.loc[j, "community_id"]:
                edge_x += [xs[i], xs[j], None]
                edge_y += [ys[i], ys[j], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.5, color="#aaa"), hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(
            size=[6 + s * 30 for s in sample["innovation_score"] / 10],
            color=sample["community_id"],
            colorscale="Turbo", showscale=True,
            colorbar=dict(title="Cluster"),
            line=dict(width=0.5, color="white"),
        ),
        text=sample["name"].str.replace("Company ", "C"),
        textposition="top center",
        textfont=dict(size=7),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Innovation: %{customdata[0]:.2f}<br>"
            "Centrality: %{customdata[1]:.3f}<br>"
            "Sector: %{customdata[2]}"
        ),
        customdata=sample[["innovation_score", "ecosystem_centrality", "sector"]].values,
    ))
    fig.update_layout(
        showlegend=False, height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Low-Density Clusters (Opportunity Flags)")
    low_density = cluster_counts[cluster_counts["density"] < median_density]
    if low_density.empty:
        st.success("No low-density clusters detected.")
    else:
        st.warning(f"{len(low_density)} cluster(s) flagged as low-density opportunities.")
        st.dataframe(low_density.rename(columns={
            "community_id": "Cluster", "companies": "# Companies",
            "avg_innovation": "Avg Innovation", "density": "Density",
        }), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predictions":
    st.title("ML Revenue Growth Predictions")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Companies Predicted", len(df), fmt="{}")
    kpi(c2, "High-Confidence (>0.80)", int(df["high_confidence"].sum()), fmt="{}")
    kpi(c3, "Avg Predicted Growth", df["predicted_growth"].mean(), delta=1.2)
    kpi(c4, "Avg Confidence Score", df["confidence_score"].mean())

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Predicted vs Actual Revenue Growth")
        fig = px.scatter(df, x="revenue_growth", y="predicted_growth",
                         color="confidence_score",
                         color_continuous_scale="RdYlGn",
                         size="innovation_score",
                         hover_name="name",
                         hover_data=["sector", "confidence_score"],
                         labels={"revenue_growth": "Actual Growth (%)",
                                 "predicted_growth": "Predicted Growth (%)"},
                         template="plotly_white")
        mn = min(df["revenue_growth"].min(), df["predicted_growth"].min())
        mx = max(df["revenue_growth"].max(), df["predicted_growth"].max())
        fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                      line=dict(dash="dash", color="gray"))
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Confidence Score Distribution")
        fig = px.histogram(df, x="confidence_score", nbins=20,
                           color="high_confidence",
                           color_discrete_map={True: "#00CC96", False: "#EF553B"},
                           labels={"confidence_score": "Confidence Score",
                                   "high_confidence": "High Confidence"},
                           template="plotly_white")
        fig.add_vline(x=0.80, line_dash="dash", line_color="black",
                      annotation_text="0.80 threshold")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("High-Growth Low-Rank Companies")
    median_growth = df["predicted_growth"].median()
    rank_75 = df["revenue_rank"].quantile(0.75)
    high_growth_low_rank = df[
        (df["predicted_growth"] > median_growth) & (df["revenue_rank"] > rank_75)
    ].sort_values("predicted_growth", ascending=False)
    st.info(f"**{len(high_growth_low_rank)}** companies with above-median predicted growth "
            f"and rank > {rank_75:.0f} (below top quartile).")
    if not high_growth_low_rank.empty:
        fig = px.scatter(high_growth_low_rank,
                         x="revenue_rank", y="predicted_growth",
                         color="sector", size="confidence_score",
                         hover_name="name",
                         labels={"revenue_rank": "Fortune 500 Rank",
                                 "predicted_growth": "Predicted Growth (%)"},
                         template="plotly_white")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction Table")
    pred_cols = ["name", "sector", "revenue_rank", "predicted_growth",
                 "revenue_growth", "confidence_score", "high_confidence"]
    st.dataframe(
        df[pred_cols].sort_values("predicted_growth", ascending=False)
        .rename(columns={
            "predicted_growth": "Predicted Growth (%)",
            "revenue_growth": "Actual Growth (%)",
            "confidence_score": "Confidence",
            "high_confidence": "High Conf.",
            "revenue_rank": "F500 Rank",
        }),
        use_container_width=True, height=350,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ROI & INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ROI & Insights":
    st.title("ROI & Business Insights")
    st.markdown("---")

    # ── ROI calculator ────────────────────────────────────────────────────────
    st.subheader("ROI Calculator")
    rc1, rc2, rc3 = st.columns(3)
    trad_hours = rc1.number_input("Traditional hours/year", 500, 5000, 2000, 100)
    sys_hours = rc1.number_input("System hours/year", 50, 1000, 200, 50)
    hourly_rate = rc2.number_input("Hourly rate ($)", 50, 500, 150, 10)
    old_decision = rc2.number_input("Old decision time (days)", 1, 60, 14)
    new_decision = rc3.number_input("New decision time (days)", 1, 30, 2)
    turnover = rc3.number_input("Turnover rate", 0.01, 0.5, 0.10, 0.01)
    kb_value = rc3.number_input("Knowledge base value ($)", 100_000, 2_000_000, 500_000, 50_000)
    sys_costs = rc1.number_input("System costs ($)", 10_000, 500_000, 100_000, 10_000)

    time_savings = (trad_hours - sys_hours) * hourly_rate
    decision_improvement = ((old_decision - new_decision) / old_decision) * 100
    knowledge_avoidance = turnover * kb_value
    # revenue impact: top quartile avg rank - bottom quartile avg rank (proxy)
    sorted_ranks = sorted(df["revenue_rank"].tolist())
    q = max(1, len(sorted_ranks) // 4)
    revenue_impact = abs(
        sum(sorted_ranks[-q:]) / q - sum(sorted_ranks[:q]) / q
    ) * 1000  # scale to dollars
    total_benefits = time_savings + revenue_impact + knowledge_avoidance
    roi_ratio = total_benefits / sys_costs if sys_costs > 0 else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    kpi(k1, "Time Savings ($)", time_savings, fmt="${:,.0f}")
    kpi(k2, "Revenue Impact ($)", revenue_impact, fmt="${:,.0f}")
    kpi(k3, "Decision Speed ↑", decision_improvement, fmt="{:.1f}%")
    kpi(k4, "Knowledge Avoidance ($)", knowledge_avoidance, fmt="${:,.0f}")
    kpi(k5, "ROI Ratio", roi_ratio, fmt="{:.2f}x")

    fig = go.Figure(go.Waterfall(
        name="ROI Breakdown",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Time Savings", "Revenue Impact", "Knowledge Avoidance", "Total Benefits"],
        y=[time_savings, revenue_impact, knowledge_avoidance, 0],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#00CC96"}},
        totals={"marker": {"color": "#636EFA"}},
    ))
    fig.update_layout(title="ROI Waterfall", template="plotly_white", height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # ── Underperformers ───────────────────────────────────────────────────────
    st.subheader("Underperformers by Sector")
    sector_avgs = df.groupby("sector")["innovation_score"].mean()
    df["sector_avg_innovation"] = df["sector"].map(sector_avgs)
    underperformers = df[df["innovation_score"] < df["sector_avg_innovation"]].copy()
    underperformers["gap"] = underperformers["sector_avg_innovation"] - underperformers["innovation_score"]
    st.warning(f"{len(underperformers)} companies are below their sector average Innovation Score.")
    fig = px.scatter(underperformers, x="innovation_score", y="gap",
                     color="sector", size="gap", hover_name="name",
                     labels={"innovation_score": "Innovation Score",
                             "gap": "Gap to Sector Average"},
                     template="plotly_white")
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # ── Acquisition targets ───────────────────────────────────────────────────
    st.subheader("Acquisition Targets")
    sector_med_centrality = df.groupby("sector")["ecosystem_centrality"].median()
    sector_med_rank = df.groupby("sector")["revenue_rank"].median()
    df["med_centrality"] = df["sector"].map(sector_med_centrality)
    df["med_rank"] = df["sector"].map(sector_med_rank)
    targets = df[
        (df["ecosystem_centrality"] > df["med_centrality"]) &
        (df["revenue_rank"] > df["med_rank"])
    ].sort_values("ecosystem_centrality", ascending=False)
    st.success(f"{len(targets)} acquisition targets identified (high centrality + low valuation).")
    st.dataframe(targets[["name", "sector", "ecosystem_centrality",
                           "revenue_rank", "innovation_score"]]
                 .rename(columns={"ecosystem_centrality": "Centrality",
                                  "revenue_rank": "F500 Rank",
                                  "innovation_score": "Innovation Score"})
                 .head(20), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CUSTOM QUERY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Custom Query":
    st.title("Custom Cypher Query Explorer")
    st.markdown("---")
    st.info("This page simulates the custom Cypher query interface. "
            "In production it connects to your Neo4j instance.")

    VALID_STARTS = ("MATCH", "CREATE", "MERGE", "RETURN", "WITH",
                    "CALL", "UNWIND", "DELETE", "SET", "REMOVE",
                    "FOREACH", "OPTIONAL")

    example_queries = {
        "Top 10 by Innovation Score":
            "MATCH (c:Company)\nRETURN c.name, c.innovation_score\nORDER BY c.innovation_score DESC\nLIMIT 10",
        "Companies in Technology sector":
            "MATCH (c:Company {sector: 'Technology'})\nRETURN c.name, c.revenue_rank\nORDER BY c.revenue_rank",
        "High centrality companies":
            "MATCH (c:Company)\nWHERE c.ecosystem_centrality > 0.2\nRETURN c.name, c.ecosystem_centrality\nORDER BY c.ecosystem_centrality DESC",
        "Acquisition targets":
            "MATCH (c:Company)\nWHERE c.ecosystem_centrality > 0.15 AND c.revenue_rank > 100\nRETURN c.name, c.sector, c.ecosystem_centrality, c.revenue_rank",
    }

    selected_example = st.selectbox("Load example query", ["— custom —"] + list(example_queries.keys()))
    default_query = example_queries.get(selected_example, "MATCH (c:Company)\nRETURN c.name, c.sector\nLIMIT 10")
    query = st.text_area("Cypher Query", value=default_query, height=120)

    col_run, col_info = st.columns([1, 4])
    run = col_run.button("▶ Execute", type="primary")

    if run:
        stripped = query.strip().upper()
        if not stripped:
            st.error("Query cannot be empty.")
        elif not any(stripped.startswith(kw) for kw in VALID_STARTS):
            st.error(f"Query must start with a valid Cypher keyword: {', '.join(VALID_STARTS)}")
        else:
            with st.spinner("Executing query…"):
                import time; time.sleep(0.4)  # simulate latency
                # Simulate results from demo dataframe
                limit = 10
                if "LIMIT" in query.upper():
                    try:
                        limit = int(query.upper().split("LIMIT")[-1].strip().split()[0])
                    except Exception:
                        limit = 10

                result_df = df[["name", "sector", "innovation_score",
                                "ecosystem_centrality", "revenue_rank",
                                "digital_maturity_index"]].head(limit)
                exec_ms = random.uniform(8, 45)

            st.success(f"Query executed in **{exec_ms:.1f} ms** · {len(result_df)} rows returned")
            st.dataframe(result_df.rename(columns={
                "innovation_score": "innovation_score",
                "ecosystem_centrality": "ecosystem_centrality",
                "revenue_rank": "revenue_rank",
                "digital_maturity_index": "digital_maturity_index",
            }), use_container_width=True)

            st.caption(f"Audit log: query executed by `anonymous` at `{datetime.now().isoformat()}`")

    st.markdown("---")
    st.subheader("Query Audit Log (Session)")
    if "audit_log" not in st.session_state:
        st.session_state.audit_log = []
    if run and query.strip():
        st.session_state.audit_log.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "user": "anonymous",
            "query_preview": query[:60].replace("\n", " ") + "…",
        })
    if st.session_state.audit_log:
        st.dataframe(pd.DataFrame(st.session_state.audit_log[::-1]),
                     use_container_width=True)
    else:
        st.caption("No queries executed yet in this session.")

# ── footer ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("Fortune 500 KG Analytics · Demo Mode")
st.sidebar.caption(f"Data: {len(df_all)} companies · {datetime.now().strftime('%Y-%m-%d')}")
