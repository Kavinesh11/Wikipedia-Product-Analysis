# Wikipedia Intelligence & Fortune 500 Analytics

A unified Streamlit analytics platform with three modules:

| Module | Description |
|---|---|
| **Wikipedia Intelligence** | Hype detection, reputation monitoring, topic clusters, forecasting |
| **Wikipedia Product Health** | Traffic trends, platform risk, changepoint detection, causal analysis |
| **Fortune 500 KG Analytics** | Knowledge graph, innovation scoring, ML predictions, ROI insights |

**Live demo:** [streamlit.io app](https://wikipedia-product-analysis.streamlit.app)

---

## Quick Start

```bash
git clone https://github.com/your-org/wikipedia-product-analysis.git
cd wikipedia-product-analysis
pip install -r requirements.txt
```

**Run the main app (Fortune 500 KG):**
```bash
streamlit run fortune500_kg/dashboard_app.py
```

**Run Wikipedia dashboards:**
```bash
streamlit run pages/1_Wikipedia_Intelligence.py
streamlit run pages/2_Wikipedia_Product_Health.py
```

**Run with Docker:**
```bash
docker-compose up -d  # opens on http://localhost:8501
```

### Environment Setup

```bash
cp .env.example .env
# Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GITHUB_TOKEN
```

---

## Wikipedia Intelligence (`pages/1_Wikipedia_Intelligence.py`)

Monitors Wikipedia article activity across 15 tracked topics.

| Page | What it shows |
|---|---|
| Overview | Hype scores, reputation risk, daily pageview area chart |
| Pageviews Analytics | Daily views by article/device, human vs bot traffic breakdown |
| Hype Detection | Composite score (view velocity + edit growth + content expansion), trending alerts |
| Reputation Monitor | Vandalism rate, edit velocity, anonymous edit %, risk level badges |
| Topic Clusters | Semantic clusters with CAGR, growth rates, emerging topic flags |
| Forecasting | 90-day per-article forecast with 95% prediction intervals |
| Knowledge Graph | PageRank, betweenness centrality, force-directed network |

**Key metrics:** hype score = `0.5×view_velocity + 0.3×edit_growth + 0.2×content_expansion`

---

## Wikipedia Product Health (`pages/2_Wikipedia_Product_Health.py`)

Analyses 2 years of platform-level traffic across desktop, mobile-web, and mobile-app.

| Page | What it shows |
|---|---|
| Overview | Total pageviews/editors/edits, HHI concentration, mobile share |
| Traffic Trends | Rolling 7/30-day MAs, platform breakdown, month-over-month growth |
| Platform Risk | HHI score, CAGR by platform, volatility, mobile decline scenarios |
| Changepoint Detection | PELT-detected structural breaks annotated on trend chart |
| Causal Analysis | ITS / DiD / Event Study effect sizes with 95% CIs and p-values |
| Seasonality | Day-of-week, monthly, 52-week cycle heatmap |
| Forecasting | Per-platform ensemble forecast (ARIMA + Prophet + ETS) with accuracy metrics |

**Risk thresholds:** HHI > 2500 = high concentration · mobile share > 70% = high dependency

---
