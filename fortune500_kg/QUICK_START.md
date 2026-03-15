# Quick Start Guide

Get up and running with Fortune 500 Knowledge Graph Analytics in 5 minutes.

## Prerequisites

- Python 3.9+
- Docker & Docker Compose
- 4GB RAM available for Docker

## Installation (5 steps)

### 1. Install Python Dependencies

```bash
cd fortune500_kg
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env if you want to change passwords
```

### 3. Start Neo4j

```bash
docker-compose up -d
```

Wait 15 seconds for Neo4j to start.

### 4. Initialize Schema

```bash
cd infrastructure
python init_schema.py
cd ..
```

### 5. Verify Installation

```bash
pytest tests/test_infrastructure.py -v
```

## Access Neo4j Browser

Open http://localhost:7474 in your browser:
- Username: `neo4j`
- Password: `fortune500password`

## Common Commands

```bash
# Start Neo4j
docker-compose up -d

# Stop Neo4j
docker-compose down

# View logs
docker logs fortune500-neo4j

# Run all tests
pytest tests/ -v

# Run property tests (100+ iterations)
pytest tests/ -m property -v

# Clear database
python -c "from infrastructure import Neo4jConnection, SchemaManager; conn = Neo4jConnection(); manager = SchemaManager(conn.get_driver()); manager.clear_database(); conn.close()"

# Reinitialize schema
cd infrastructure && python init_schema.py && cd ..
```

## Example Cypher Queries

Try these in Neo4j Browser:

```cypher
// Create a sample company
CREATE (c:Company {
    id: 'COMP001',
    name: 'Tech Corp',
    sector: 'Technology',
    revenue_rank: 1,
    employee_count: 100000,
    github_org: 'techcorp',
    created_at: datetime(),
    updated_at: datetime()
})

// Create a sample repository
CREATE (r:Repository {
    id: 'REPO001',
    name: 'awesome-project',
    stars: 5000,
    forks: 1000,
    contributors: 50,
    created_at: datetime()
})

// Create ownership relationship
MATCH (c:Company {id: 'COMP001'})
MATCH (r:Repository {id: 'REPO001'})
CREATE (c)-[:OWNS]->(r)

// Query companies and their repositories
MATCH (c:Company)-[:OWNS]->(r:Repository)
RETURN c.name, r.name, r.stars
ORDER BY r.stars DESC
```

## Troubleshooting

**Can't connect to Neo4j?**
```bash
docker ps | grep neo4j  # Check if running
docker logs fortune500-neo4j  # Check logs
```

**Port already in use?**
```bash
# Change ports in docker-compose.yml
# Or stop conflicting service
```

**Tests failing?**
```bash
# Ensure Neo4j is running
docker ps

# Ensure test instance is running
docker ps | grep neo4j-test

# Restart if needed
docker-compose restart
```

## Next Steps

1. ✅ Infrastructure is ready
2. 📝 Implement data ingestion pipeline (Task 2)
3. 📊 Implement analytics engine (Task 4)
4. 🤖 Implement ML models (Task 9)
5. 📈 Implement dashboards (Task 18)

## Documentation

- [Installation Guide](INSTALLATION.md) - Detailed setup instructions
- [Infrastructure README](infrastructure/README.md) - Schema and API docs
- [Task 1 Summary](TASK_1_SUMMARY.md) - Implementation details

## Support

- Check Neo4j logs: `docker logs fortune500-neo4j`
- Verify setup: `python verify_setup.py`
- Review test output: `pytest tests/ -v`
