# Deployment Scripts

This directory contains scripts for deploying, managing, and maintaining the Wikipedia Intelligence System.

## Scripts Overview

### startup.py

Main application startup script that handles initialization, health checks, and application launch.

**Usage:**
```bash
python scripts/startup.py
```

**Features:**
- Loads and validates configuration
- Runs health checks on all components
- Initializes required directories
- Runs database migrations
- Starts the dashboard application

**Environment Variables:**
- `ENVIRONMENT` - Environment to run (development/staging/production)

### deploy.sh

Quick deployment script for automated deployment.

**Usage:**
```bash
# Docker deployment
./scripts/deploy.sh docker

# Manual deployment
./scripts/deploy.sh development
./scripts/deploy.sh staging
./scripts/deploy.sh production

# Skip database backup
./scripts/deploy.sh production true
```

**Features:**
- Checks prerequisites
- Builds Docker images (if using Docker)
- Creates backups (production only)
- Runs migrations
- Validates configuration
- Starts application

### health_check.py

Comprehensive health check script for monitoring system components.

**Usage:**
```bash
# Run all health checks
python scripts/health_check.py

# Run specific check
python scripts/health_check.py --check database
python scripts/health_check.py --check redis
python scripts/health_check.py --check api

# Output as JSON
python scripts/health_check.py --json

# Check specific environment
python scripts/health_check.py --environment production
```

**Checks:**
- Database connectivity and metrics
- Redis connectivity and metrics
- Wikimedia API availability
- Disk space usage
- Log file sizes

**Exit Codes:**
- 0: All checks passed
- 1: One or more checks failed

### migrate_data.py

Database migration script for schema updates and data transformations.

**Usage:**
```bash
# Run all pending migrations
python scripts/migrate_data.py

# Run migrations for specific environment
python scripts/migrate_data.py --environment production

# Migrate to specific version
python scripts/migrate_data.py --target-version 3

# Skip backup
python scripts/migrate_data.py --skip-backup
```

**Features:**
- Creates database backups before migration
- Runs incremental migrations
- Tracks schema version
- Supports rollback (via backups)

### init_db.sql

SQL script for initializing the database with required extensions, schemas, and tables.

**Usage:**
```bash
# Run directly with psql
psql -U wiki_user -d wikipedia_intelligence -f scripts/init_db.sql

# Or via Docker
docker exec -i wikipedia-intelligence-db psql -U wiki_user -d wikipedia_intelligence < scripts/init_db.sql
```

**Creates:**
- PostgreSQL extensions (uuid-ossp, pg_trgm)
- Application schema
- Audit log table
- System health table
- Runtime configuration table

## Common Tasks

### Initial Setup

```bash
# 1. Initialize database
psql -U wiki_user -d wikipedia_intelligence -f scripts/init_db.sql

# 2. Run migrations
alembic upgrade head

# 3. Start application
python scripts/startup.py
```

### Health Monitoring

```bash
# Quick health check
python scripts/health_check.py

# Continuous monitoring (every 5 minutes)
watch -n 300 python scripts/health_check.py

# Add to cron for automated checks
*/5 * * * * /path/to/venv/bin/python /path/to/scripts/health_check.py --json >> /var/log/health_checks.log
```

### Deployment

```bash
# Development
./scripts/deploy.sh development

# Production with Docker
./scripts/deploy.sh docker

# Production manual
./scripts/deploy.sh production
```

### Database Maintenance

```bash
# Create backup
pg_dump -U wiki_user -d wikipedia_intelligence -f backup_$(date +%Y%m%d).sql

# Run migrations
python scripts/migrate_data.py --environment production

# Restore from backup
psql -U wiki_user -d wikipedia_intelligence -f backup_20240101.sql
```

## Automation

### Systemd Service

Create `/etc/systemd/system/wikipedia-intelligence.service`:

```ini
[Unit]
Description=Wikipedia Intelligence System
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=wiki
WorkingDirectory=/opt/wikipedia-intelligence
Environment="ENVIRONMENT=production"
EnvironmentFile=/opt/wikipedia-intelligence/.env
ExecStart=/opt/wikipedia-intelligence/venv/bin/python scripts/startup.py
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Cron Jobs

Add to crontab for automated tasks:

```bash
# Health checks every 5 minutes
*/5 * * * * cd /opt/wikipedia-intelligence && venv/bin/python scripts/health_check.py --json >> logs/health_checks.log 2>&1

# Database backup daily at 2 AM
0 2 * * * pg_dump -U wiki_user -d wikipedia_intelligence -f /backups/wiki_$(date +\%Y\%m\%d).sql

# Log rotation weekly
0 0 * * 0 find /opt/wikipedia-intelligence/logs -name "*.log" -mtime +7 -delete
```

## Troubleshooting

### Script Permissions

Make scripts executable:
```bash
chmod +x scripts/deploy.sh
chmod +x scripts/startup.py
chmod +x scripts/health_check.py
chmod +x scripts/migrate_data.py
```

### Python Path Issues

If scripts can't find modules:
```bash
export PYTHONPATH=/path/to/wikipedia-intelligence/src:$PYTHONPATH
```

### Database Connection Issues

Check connection manually:
```bash
psql -U wiki_user -d wikipedia_intelligence -c "SELECT 1"
```

### Redis Connection Issues

Check connection manually:
```bash
redis-cli -h localhost -p 6379 ping
```

## Security Notes

1. **Never commit** `.env` files with real credentials
2. **Restrict permissions** on scripts containing sensitive data
3. **Use environment variables** for production credentials
4. **Rotate credentials** regularly
5. **Audit script execution** in production

## Script Dependencies

All scripts require:
- Python 3.11+
- PyYAML
- SQLAlchemy
- Redis client
- Requests

Install with:
```bash
pip install -r requirements.txt
```
