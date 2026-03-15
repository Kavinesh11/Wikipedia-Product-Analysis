#!/bin/bash
# Setup script for Fortune 500 Knowledge Graph Analytics

set -e

echo "=========================================="
echo "Fortune 500 Knowledge Graph Setup"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file from example if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "✓ .env file created. Please update with your credentials."
fi

# Start Neo4j containers
echo ""
echo "Starting Neo4j containers..."
docker-compose up -d

# Wait for Neo4j to be ready
echo ""
echo "Waiting for Neo4j to be ready..."
sleep 10

# Check if Neo4j is running
if docker ps | grep -q fortune500-neo4j; then
    echo "✓ Neo4j main instance is running"
else
    echo "✗ Neo4j main instance failed to start"
    exit 1
fi

if docker ps | grep -q fortune500-neo4j-test; then
    echo "✓ Neo4j test instance is running"
else
    echo "✗ Neo4j test instance failed to start"
    exit 1
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Initialize schema
echo ""
echo "Initializing Neo4j schema..."
cd infrastructure
python init_schema.py
cd ..

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Neo4j Browser (main): http://localhost:7474"
echo "Neo4j Browser (test): http://localhost:7475"
echo ""
echo "To run tests:"
echo "  pytest tests/"
echo ""
echo "To run property-based tests:"
echo "  pytest tests/ -m property"
echo ""
