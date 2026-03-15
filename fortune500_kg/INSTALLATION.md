# Installation Guide

## Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose
- Git

## Step-by-Step Installation

### 1. Clone or Navigate to Project Directory

```bash
cd fortune500_kg
```

### 2. Create Python Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- neo4j (Python driver)
- pandas, numpy (data processing)
- pytest, hypothesis (testing)
- And other required packages

### 4. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your preferred editor
# Update passwords if needed
```

### 5. Start Neo4j with Docker

```bash
# Start both main and test Neo4j instances
docker-compose up -d

# Verify containers are running
docker ps | grep fortune500-neo4j

# Check logs if needed
docker logs fortune500-neo4j
```

**Wait 10-15 seconds** for Neo4j to fully start up.

### 6. Initialize Graph Schema

```bash
cd infrastructure
python init_schema.py
cd ..
```

You should see output like:
```
============================================================
Fortune 500 Knowledge Graph - Schema Initialization
============================================================

1. Connecting to Neo4j...
✓ Connected to Neo4j successfully

2. Initializing schema...
✓ Constraints created successfully
✓ Indexes created successfully
✓ Schema initialization complete

3. Validating schema...
  ✓ constraints_exist: True
  ✓ indexes_exist: True
  ...

✓ All schema elements validated successfully
```

### 7. Verify Installation

```bash
# Run unit tests
pytest tests/test_infrastructure.py -v

# Run property-based tests (takes longer)
pytest tests/test_schema_properties.py -m property -v
```

## Quick Setup Script (Linux/Mac)

For convenience, you can use the setup script:

```bash
chmod +x setup.sh
./setup.sh
```

## Accessing Neo4j Browser

Once Neo4j is running, you can access the browser interface:

- **Main Instance**: http://localhost:7474
  - Username: `neo4j`
  - Password: `fortune500password` (or as configured in .env)

- **Test Instance**: http://localhost:7475
  - Username: `neo4j`
  - Password: `testpassword` (or as configured in .env)

## Troubleshooting

### Docker Issues

**Problem**: Docker containers won't start

**Solution**:
```bash
# Check Docker is running
docker --version

# Check for port conflicts
# On Windows:
netstat -ano | findstr "7474"
netstat -ano | findstr "7687"

# On Linux/Mac:
lsof -i :7474
lsof -i :7687

# If ports are in use, stop conflicting services or change ports in docker-compose.yml
```

### Python Module Issues

**Problem**: `ModuleNotFoundError: No module named 'neo4j'`

**Solution**:
```bash
# Ensure you're in the virtual environment
# Then reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Neo4j Connection Issues

**Problem**: Cannot connect to Neo4j

**Solution**:
```bash
# 1. Check containers are running
docker ps

# 2. Check container logs
docker logs fortune500-neo4j

# 3. Restart containers
docker-compose restart

# 4. If still failing, remove and recreate
docker-compose down -v
docker-compose up -d
```

### Schema Initialization Fails

**Problem**: Schema initialization script fails

**Solution**:
```bash
# 1. Ensure Neo4j is fully started (wait 15 seconds after docker-compose up)
# 2. Check connection settings in .env
# 3. Try clearing and reinitializing:

python -c "
from infrastructure import Neo4jConnection, SchemaManager
conn = Neo4jConnection()
driver = conn.get_driver()
manager = SchemaManager(driver)
manager.clear_database()
manager.initialize_schema()
conn.close()
"
```

## Uninstallation

To completely remove the installation:

```bash
# Stop and remove containers
docker-compose down -v

# Remove virtual environment
# On Windows:
rmdir /s venv
# On Linux/Mac:
rm -rf venv

# Remove .env file if desired
rm .env
```

## Next Steps

After successful installation:

1. Review the [Infrastructure README](infrastructure/README.md) for schema details
2. Explore the Cypher schema files in `infrastructure/schema/`
3. Run the test suite to verify everything works
4. Start implementing the data ingestion pipeline (Task 2)

## Support

For issues or questions:
- Check the [Infrastructure README](infrastructure/README.md)
- Review Neo4j logs: `docker logs fortune500-neo4j`
- Verify all prerequisites are installed
- Ensure Docker has sufficient resources allocated
