<div align="center">
  
# Wikipedia Intelligence & Fortune 500 Analytics

A unified Streamlit analytics platform with three modules:
</div>

| Module | Description |
|---|---|
| **Wikipedia Intelligence** | Hype detection, reputation monitoring, topic clusters, forecasting |
| **Wikipedia Product Health** | Traffic trends, platform risk, changepoint detection, causal analysis |
| **Fortune 500 KG Analytics** | Knowledge graph, innovation scoring, ML predictions, ROI insights |

**Live demo:** [streamlit.io app 1](https://wikipedia-health.streamlit.app/)
[streamlit.io app 2](https://wiki-intelligence.streamlit.app/)
[streamlit.io app 3](https://fortune500wiki.streamlit.app/)

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

## Fortune 500 KG Analytics (`fortune500_kg/`)

Ingests Fortune 500 company data into a Neo4j knowledge graph, runs graph algorithms, trains ML models, and surfaces executive insights.

### Dashboard Pages (`fortune500_kg/dashboard_app.py`)

| Page | What it shows |
|---|---|
| Overview | KPI row, innovation score distribution, digital maturity by sector, trend line |
| Leaderboard | Top companies by innovation score, decile distribution, innovation vs revenue rank scatter |
| Sector Analysis | Cross-sector bar/radar charts, trend lines, heatmap, inter-sector % differences |
| Network & Clusters | Louvain communities, network density, force-directed graph (top 60 by centrality) |
| Predictions | Predicted vs actual revenue growth, confidence distribution, high-growth low-rank table |
| ROI & Insights | Interactive ROI calculator, waterfall chart, underperformers, acquisition targets |
| Custom Query | Cypher query interface with syntax validation and audit logging |

### Core Metrics

- **Innovation Score** — `(stars + forks) / employee_count`, normalised 0–10, decile ranked
- **Digital Maturity Index** — `(stars + forks + contributors) / revenue_rank`
- **Ecosystem Centrality** — betweenness + PageRank composite
- **Acquisition targets** — high centrality + below-median revenue rank

### Module Structure

```
fortune500_kg/
├── dashboard_app.py          # Streamlit multi-page dashboard
├── analytics_engine.py       # Innovation Score, PageRank, Louvain, correlation, sector analysis
├── data_ingestion_pipeline.py# Crawl4AI parser + GitHub API with rate limiting
├── data_models.py            # All dataclasses (Company, MetricRecord, ExecutiveReport, …)
├── dashboard_service.py      # Leaderboard, network graph, heatmap, Bloom overlay
├── insight_generator.py      # Underperformers, recommendations, acquisition targets, ROI
├── predictive_model.py       # ML revenue growth prediction + validation
├── metrics_exporter.py       # CSV / JSON / Tableau / Power BI export
├── performance_monitor.py    # Execution time, memory, throughput, health dashboard
├── error_handler.py          # Retry decorator, failure rate tracking
├── infrastructure/           # Neo4j schema + Cypher migration scripts
├── templates/                # Jinja2 HTML report templates
└── tests/                    # 30 test modules, 362 passing (pytest + Hypothesis)
```

---

## Testing

```bash
pytest                          # all tests
pytest tests/ -m property       # Hypothesis property-based tests only
pytest fortune500_kg/tests/ -v  # Fortune 500 KG tests
```

## Project Structure

```
.
├── fortune500_kg/              # Fortune 500 KG Analytics module
├── wikipedia_health/           # Wikipedia Product Health analysis engine
│   ├── analysis_system.py      # Main orchestrator
│   ├── data_acquisition/       # Wikimedia API client
│   ├── time_series/            # Changepoint detection, decomposition, forecasting
│   ├── statistical_validation/ # Hypothesis testing, confidence intervals, effect sizes
│   ├── causal_inference/       # ITS, DiD, Event Study, Synthetic Control
│   └── visualization/          # Plotly dashboard components
├── pages/
│   ├── 1_Wikipedia_Intelligence.py   # Wikipedia Intelligence dashboard
│   └── 2_Wikipedia_Product_Health.py # Product Health dashboard
├── streamlit_app.py            # Streamlit Cloud entry point
├── config/                     # YAML configuration files
├── docs/                       # Extended documentation
└── tests/                      # Root-level test suite
```

## License

MIT — see [LICENSE](LICENSE) for details.
