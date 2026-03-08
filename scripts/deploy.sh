#!/bin/bash
# Quick Deployment Script for Wikipedia Intelligence System

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 is not installed"
        return 1
    fi
    log_info "$1 is installed"
    return 0
}

# Parse arguments
ENVIRONMENT=${1:-development}
SKIP_BACKUP=${2:-false}

log_info "Starting deployment for environment: $ENVIRONMENT"

# Check prerequisites
log_info "Checking prerequisites..."

if [ "$ENVIRONMENT" = "docker" ]; then
    check_command docker || exit 1
    check_command docker-compose || exit 1
else
    check_command python3 || exit 1
    check_command psql || exit 1
    check_command redis-cli || exit 1
fi

# Load environment variables
if [ -f .env ]; then
    log_info "Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
else
    log_warn ".env file not found, using defaults"
fi

# Docker deployment
if [ "$ENVIRONMENT" = "docker" ]; then
    log_info "Deploying with Docker Compose..."
    
    # Check if .env exists
    if [ ! -f .env ]; then
        log_warn "Creating .env from template"
        cp .env.docker .env
        log_warn "Please edit .env with your settings before running in production!"
    fi
    
    # Build images
    log_info "Building Docker images..."
    docker-compose build
    
    # Start services
    log_info "Starting services..."
    docker-compose up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 10
    
    # Check health
    if docker-compose ps | grep -q "unhealthy"; then
        log_error "Some services are unhealthy"
        docker-compose ps
        exit 1
    fi
    
    log_info "All services are running"
    log_info "Dashboard available at: http://localhost:8501"
    
    exit 0
fi

# Manual deployment
log_info "Deploying manually for environment: $ENVIRONMENT"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    log_info "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
log_info "Installing dependencies..."
pip install -r requirements.txt --quiet

# Create necessary directories
log_info "Creating directories..."
mkdir -p logs data output backups

# Check database connection
log_info "Checking database connection..."
if ! psql -U ${POSTGRES_USER:-wiki_user} -d ${POSTGRES_DB:-wikipedia_intelligence} -c "SELECT 1" > /dev/null 2>&1; then
    log_error "Cannot connect to database"
    log_info "Please ensure PostgreSQL is running and credentials are correct"
    exit 1
fi
log_info "Database connection: OK"

# Check Redis connection
log_info "Checking Redis connection..."
if ! redis-cli -h ${REDIS_HOST:-localhost} -p ${REDIS_PORT:-6379} ping > /dev/null 2>&1; then
    log_error "Cannot connect to Redis"
    log_info "Please ensure Redis is running"
    exit 1
fi
log_info "Redis connection: OK"

# Backup database (unless skipped)
if [ "$SKIP_BACKUP" != "true" ] && [ "$ENVIRONMENT" = "production" ]; then
    log_info "Creating database backup..."
    BACKUP_FILE="backups/backup_$(date +%Y%m%d_%H%M%S).sql"
    pg_dump -U ${POSTGRES_USER:-wiki_user} -d ${POSTGRES_DB:-wikipedia_intelligence} -f $BACKUP_FILE
    log_info "Backup created: $BACKUP_FILE"
fi

# Run database migrations
log_info "Running database migrations..."
if ! alembic upgrade head; then
    log_error "Database migration failed"
    exit 1
fi
log_info "Database migrations: OK"

# Run data migrations if needed
if [ -f "scripts/migrate_data.py" ]; then
    log_info "Running data migrations..."
    python scripts/migrate_data.py --environment $ENVIRONMENT --skip-backup
fi

# Validate configuration
log_info "Validating configuration..."
if ! python -c "
import sys
sys.path.insert(0, 'src')
from utils.config import load_config
config = load_config('$ENVIRONMENT')
print('Configuration valid')
" 2>/dev/null; then
    log_error "Configuration validation failed"
    exit 1
fi
log_info "Configuration: OK"

# Start application
log_info "Starting application..."
log_info "Environment: $ENVIRONMENT"
log_info "Dashboard will be available at: http://localhost:8501"

# Run startup script
python scripts/startup.py

log_info "Deployment completed successfully!"
