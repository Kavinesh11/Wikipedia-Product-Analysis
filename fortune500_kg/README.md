# Fortune 500 Knowledge Graph Analytics

A comprehensive business intelligence platform that leverages Neo4j graph database technology to analyze Fortune 500 companies through innovation metrics, network centrality analysis, digital maturity assessment, and predictive analytics.

## Project Structure

```
fortune500_kg/
├── infrastructure/          # Neo4j setup and configuration
│   ├── schema/             # Cypher scripts for graph schema
│   └── docker/             # Docker configuration for Neo4j
├── ingestion/              # Data ingestion pipeline
├── analytics/              # Analytics engine and algorithms
├── ml/                     # Machine learning models
├── insights/               # Insight generation
├── dashboard/              # Visualization and dashboards
└── tests/                  # Test suite
```

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Neo4j 5.x with Graph Data Science (GDS) library

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Neo4j with Docker:
```bash
docker-compose up -d
```

3. Initialize the graph schema:
```bash
python -m fortune500_kg.infrastructure.init_schema
```

## Testing

Run all tests:
```bash
pytest tests/
```

Run property-based tests:
```bash
pytest tests/ -m property
```
