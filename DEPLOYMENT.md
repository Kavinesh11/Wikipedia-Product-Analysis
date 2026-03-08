# Deployment Guide

## Overview

This guide covers deploying the Wikipedia Intelligence System in different environments using Docker, Docker Compose, or manual installation.

## Prerequisites

### For Docker Deployment
- Docker 20.10 or later
- Docker Compose 2.0 or later
- 4GB RAM minimum (8GB recommended)
- 20GB disk space

### For Manual Deployment
- Python 3.11 or later
- PostgreSQL 15 or later
- Redis 7 or later
- 4GB RAM minimum (8GB recommended)
- 20GB disk space

## Quick Start with Docker Compose

### 1. Clone Repository

```bash
git clone <repository-url>
cd wikipedia-intelligence-system
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.docker .env

# Edit .env with your settings
nano .env
```

**Important**: Change the default passwords in production!

### 3. Start Services

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f app
```

### 4. Access Dashboard

Open your browser to: `http://localhost:8501`

### 5. Stop Services

```bash
docker-compose down

# To remove volumes (WARNING: deletes all data)
docker-compose down -v
```

## Manual Deployment

### 1. Install Dependencies

#### PostgreSQL

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install postgresql-15 postgresql-client-15

# macOS
brew install postgresql@15

# Start PostgreSQL
sudo systemctl start postgresql  # Linux
brew services start postgresql@15  # macOS
```

#### Redis

```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Start Redis
sudo systemctl start redis  # Linux
brew services start redis  # macOS
```

#### Python Dependencies

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Database

```bash
# Create database and user
sudo -u postgres psql

CREATE DATABASE wikipedia_intelligence;
CREATE USER wiki_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE wikipedia_intelligence TO wiki_user;
\q
```

### 3. Initialize Database

```bash
# Run initialization script
psql -U wiki_user -d wikipedia_intelligence -f scripts/init_db.sql

# Run migrations
alembic upgrade head
```

### 4. Configure Application

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

Set the following variables:
```
ENVIRONMENT=production
POSTGRES_HOST=localhost
POSTGRES_DB=wikipedia_intelligence
POSTGRES_USER=wiki_user
POSTGRES_PASSWORD=your_secure_password
REDIS_HOST=localhost
```

### 5. Start Application

```bash
# Run startup script
python scripts/startup.py

# Or start dashboard directly
streamlit run src/visualization/dashboard.py --server.port 8501
```

## Production Deployment

### Security Checklist

- [ ] Change all default passwords
- [ ] Enable SSL/TLS for database connections
- [ ] Configure firewall rules
- [ ] Set up log rotation
- [ ] Enable monitoring and alerting
- [ ] Configure backup strategy
- [ ] Restrict network access to services
- [ ] Use secrets management (e.g., HashiCorp Vault)
- [ ] Enable audit logging
- [ ] Configure rate limiting

### Environment Configuration

Create production configuration:

```bash
# Set environment
export ENVIRONMENT=production

# Database (use strong passwords!)
export POSTGRES_HOST=prod-db.example.com
export POSTGRES_DB=wikipedia_intelligence
export POSTGRES_USER=wiki_prod_user
export POSTGRES_PASSWORD=<strong-password>

# Redis
export REDIS_HOST=prod-redis.example.com

# Alerts
export ALERT_EMAIL_ENABLED=true
export ALERT_WEBHOOK_URL=https://your-webhook-url.com
```

### SSL/TLS Configuration

For production, enable SSL:

```yaml
# config/production.yaml
security:
  enable_ssl: true
  ssl_cert_path: "/path/to/cert.pem"
  ssl_key_path: "/path/to/key.pem"
```

### Reverse Proxy Setup (Nginx)

```nginx
server {
    listen 80;
    server_name intelligence.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name intelligence.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Systemd Service (Linux)

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
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable wikipedia-intelligence
sudo systemctl start wikipedia-intelligence
sudo systemctl status wikipedia-intelligence
```

## Monitoring and Health Checks

### Health Check Endpoints

The application exposes health check endpoints:

- **Application Health**: `http://localhost:8501/_stcore/health`
- **Database Health**: Check via startup script logs
- **Redis Health**: Check via startup script logs

### Monitoring with Docker

```bash
# Check container health
docker ps

# View container logs
docker logs wikipedia-intelligence-app

# Check resource usage
docker stats
```

### Log Files

Logs are stored in the `logs/` directory:

- `app.log` - Application logs
- `app_dev.log` - Development logs
- `app_staging.log` - Staging logs
- `app_production.log` - Production logs

## Backup and Recovery

### Database Backup

```bash
# Manual backup
pg_dump -U wiki_user -d wikipedia_intelligence -f backup_$(date +%Y%m%d).sql

# Automated backup (add to crontab)
0 2 * * * pg_dump -U wiki_user -d wikipedia_intelligence -f /backups/wiki_$(date +\%Y\%m\%d).sql
```

### Redis Backup

Redis automatically saves snapshots to disk. Configure in `redis.conf`:

```
save 900 1
save 300 10
save 60 10000
```

### Restore from Backup

```bash
# Restore database
psql -U wiki_user -d wikipedia_intelligence -f backup_20240101.sql

# Restore Redis
redis-cli SHUTDOWN SAVE
cp backup.rdb /var/lib/redis/dump.rdb
redis-server
```

## Scaling

### Horizontal Scaling

The system supports horizontal scaling:

1. **Database**: Use PostgreSQL replication (primary-replica)
2. **Cache**: Use Redis Cluster for distributed caching
3. **Application**: Run multiple app instances behind a load balancer
4. **Workers**: Scale crawler and ETL workers independently

### Load Balancer Configuration

```nginx
upstream wiki_intelligence {
    least_conn;
    server app1.example.com:8501;
    server app2.example.com:8501;
    server app3.example.com:8501;
}

server {
    listen 443 ssl http2;
    server_name intelligence.example.com;

    location / {
        proxy_pass http://wiki_intelligence;
        # ... other proxy settings
    }
}
```

## Troubleshooting

### Application Won't Start

1. Check logs: `docker-compose logs app` or `tail -f logs/app.log`
2. Verify database connection: `psql -U wiki_user -d wikipedia_intelligence`
3. Verify Redis connection: `redis-cli ping`
4. Check configuration: `python scripts/startup.py` (will validate config)

### Database Connection Errors

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check connection
psql -U wiki_user -d wikipedia_intelligence

# Check pg_hba.conf for access rules
sudo nano /etc/postgresql/15/main/pg_hba.conf
```

### Redis Connection Errors

```bash
# Check Redis is running
sudo systemctl status redis

# Check connection
redis-cli ping

# Check Redis configuration
redis-cli CONFIG GET bind
```

### Performance Issues

1. Check resource usage: `docker stats` or `htop`
2. Review slow queries in PostgreSQL logs
3. Check Redis memory usage: `redis-cli INFO memory`
4. Increase connection pool sizes in config
5. Enable query caching

### Out of Memory

1. Increase Docker memory limits in `docker-compose.yml`
2. Reduce `max_workers` in configuration
3. Reduce `pool_size` for database connections
4. Enable Redis memory eviction policies

## Maintenance

### Regular Tasks

- **Daily**: Check logs for errors
- **Weekly**: Review disk space usage
- **Monthly**: Update dependencies
- **Quarterly**: Review and rotate credentials

### Updates

```bash
# Pull latest code
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Run migrations
alembic upgrade head

# Restart application
docker-compose restart app
# or
sudo systemctl restart wikipedia-intelligence
```

## Support

For issues and questions:
- Check logs first
- Review configuration documentation
- Consult API documentation
- Contact system administrator
