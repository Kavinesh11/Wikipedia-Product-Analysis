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
