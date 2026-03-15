# Neo4j Infrastructure

This directory contains the Neo4j infrastructure setup for the Fortune 500 Knowledge Graph Analytics system.

## Directory Structure

```
infrastructure/
├── schema/                          # Cypher schema scripts
│   ├── 01_create_constraints.cypher # Uniqueness constraints
│   ├── 02_create_indexes.cypher     # Performance indexes
│   └── 03_schema_documentation.cypher # Schema documentation
├── connection.py                    # Neo4j connection management
├── schema_manager.py                # Schema initialization and validation
└── init_schema.py                   # Schema initialization script
```

## Graph Schema

### Node Types

#### Company
Represents a Fortune 500 company with business and GitHub metrics.

**Properties:**
- `id` (String, UNIQUE): Unique identifier
- `name` (String): Company name
- `sector` (String, INDEXED): Industry sector
- `revenue_rank` (Integer, INDEXED): Fortune 500 rank
- `employee_count` (Integer): Number of employees
- `github_org` (String): GitHub organization name
- `created_at` (DateTime): Node creation timestamp
- `updated_at` (DateTime): Last update timestamp

#### Repository
Represents a GitHub repository owned or used by companies.

**Properties:**
- `id` (String, UNIQUE): GitHub repository ID
- `name` (String): Repository name
- `stars` (Integer): GitHub stars count
- `forks` (Integer): GitHub forks count
- `contributors` (Integer): Number of contributors
- `created_at` (DateTime): Repository creation timestamp

#### Sector
Represents an industry sector with aggregated metrics.

**Properties:**
- `id` (String, UNIQUE): Sector identifier
- `name` (String): Sector name
- `avg_innovation_score` (Float): Average innovation score
- `avg_digital_maturity` (Float): Average digital maturity

### Relationship Types

#### OWNS
`(:Company)-[:OWNS]->(:Repository)`

Indicates a company owns a GitHub repository.

**Properties:** None

#### PARTNERS_WITH
`(:Company)-[:PARTNERS_WITH]->(:Company)`

Indicates a partnership between two companies.

**Properties:**
- `since` (Date): Partnership start date
- `partnership_type` (String): Type of partnership

#### ACQUIRED
`(:Company)-[:ACQUIRED]->(:Company)`

Indicates one company acquired another.

**Properties:**
- `date` (Date): Acquisition date
- `amount` (Float): Acquisition amount in USD

#### BELONGS_TO
`(:Company)-[:BELONGS_TO]->(:Sector)`

Indicates a company belongs to an industry sector.

**Properties:** None

#### DEPENDS_ON
`(:Company)-[:DEPENDS_ON]->(:Repository)`

Indicates a company has a technology dependency on a repository.

**Properties:**
- `dependency_type` (String): Type of dependency (e.g., "direct", "transitive")

## Indexes

The following indexes are created for query performance:

### Constraints (with automatic indexes)
- `company_id_unique`: Unique constraint on Company.id
- `repository_id_unique`: Unique constraint on Repository.id
- `sector_id_unique`: Unique constraint on Sector.id

### Performance Indexes
- `company_sector_idx`: Index on Company.sector
- `company_revenue_rank_idx`: Index on Company.revenue_rank
- `company_name_idx`: Index on Company.name
- `repository_name_idx`: Index on Repository.name
- `repository_stars_idx`: Index on Repository.stars
- `sector_name_idx`: Index on Sector.name
- `company_sector_revenue_idx`: Composite index on (Company.sector, Company.revenue_rank)

## Usage

### Initialize Schema

```python
from infrastructure import Neo4jConnection, SchemaManager

# Connect to Neo4j
conn = Neo4jConnection()
driver = conn.get_driver()

# Initialize schema
schema_manager = SchemaManager(driver)
schema_manager.initialize_schema()

# Validate schema
validation = schema_manager.validate_schema()
print(validation)

# Close connection
conn.close()
```

### Command Line

```bash
# Initialize schema
cd infrastructure
python init_schema.py
```

### Docker Setup

```bash
# Start Neo4j containers
docker-compose up -d

# Check container status
docker ps | grep fortune500-neo4j

# View logs
docker logs fortune500-neo4j

# Stop containers
docker-compose down
```

## Configuration

Configuration is managed through environment variables. Copy `.env.example` to `.env` and update:

```bash
# Main Neo4j instance
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=fortune500password

# Test Neo4j instance
NEO4J_TEST_URI=bolt://localhost:7688
NEO4J_TEST_USER=neo4j
NEO4J_TEST_PASSWORD=testpassword
```

## Testing

The infrastructure includes both unit tests and property-based tests:

```bash
# Run all infrastructure tests
pytest tests/test_infrastructure.py -v

# Run property-based tests
pytest tests/test_schema_properties.py -m property -v

# Run with coverage
pytest tests/ --cov=infrastructure --cov-report=html
```

## Neo4j Browser Access

- **Main Instance**: http://localhost:7474
  - Username: neo4j
  - Password: fortune500password

- **Test Instance**: http://localhost:7475
  - Username: neo4j
  - Password: testpassword

## Troubleshooting

### Connection Issues

If you can't connect to Neo4j:

1. Check if containers are running:
   ```bash
   docker ps | grep neo4j
   ```

2. Check container logs:
   ```bash
   docker logs fortune500-neo4j
   ```

3. Verify ports are not in use:
   ```bash
   lsof -i :7474
   lsof -i :7687
   ```

### Schema Issues

If schema initialization fails:

1. Clear the database:
   ```python
   schema_manager.clear_database()
   ```

2. Re-initialize:
   ```python
   schema_manager.initialize_schema()
   ```

3. Validate:
   ```python
   validation = schema_manager.validate_schema()
   ```

### Memory Issues

If Neo4j runs out of memory, adjust heap size in `docker-compose.yml`:

```yaml
environment:
  - NEO4J_dbms_memory_heap_max__size=4G  # Increase as needed
  - NEO4J_dbms_memory_pagecache_size=2G
```

## Performance Tuning

For production deployments, consider:

1. **Heap Size**: Increase based on dataset size
2. **Page Cache**: Set to ~50% of available RAM
3. **Indexes**: Monitor query performance and add indexes as needed
4. **GDS Library**: Ensure Graph Data Science library is properly installed

## References

- [Neo4j Documentation](https://neo4j.com/docs/)
- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/)
- [Graph Data Science Library](https://neo4j.com/docs/graph-data-science/current/)
