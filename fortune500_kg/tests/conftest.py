"""Pytest configuration and fixtures for testing."""

import os
import pytest
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@pytest.fixture(scope='session')
def neo4j_test_uri():
    """Neo4j test instance URI."""
    return os.getenv('NEO4J_TEST_URI', 'neo4j://localhost:7688')


@pytest.fixture(scope='session')
def neo4j_test_credentials():
    """Neo4j test instance credentials."""
    return (
        os.getenv('NEO4J_TEST_USER', 'neo4j'),
        os.getenv('NEO4J_TEST_PASSWORD', 'testpassword')
    )


@pytest.fixture(scope='session')
def neo4j_test_driver(neo4j_test_uri, neo4j_test_credentials):
    """
    Create a Neo4j driver for the test instance.
    This fixture has session scope and is shared across all tests.
    """
    user, password = neo4j_test_credentials
    driver = GraphDatabase.driver(neo4j_test_uri, auth=(user, password))
    
    # Verify connectivity
    try:
        driver.verify_connectivity()
    except Exception as e:
        pytest.skip(f"Neo4j test instance not available: {e}")
    
    yield driver
    
    driver.close()


@pytest.fixture(scope='function')
def clean_neo4j_db(neo4j_test_driver):
    """
    Clean the Neo4j test database before each test.
    This fixture has function scope and runs before each test.
    """
    with neo4j_test_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    
    yield neo4j_test_driver
    
    # Clean up after test
    with neo4j_test_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


@pytest.fixture
def sample_company_data():
    """Sample company data for testing."""
    return {
        'id': 'COMP001',
        'name': 'Tech Corp',
        'sector': 'Technology',
        'revenue_rank': 1,
        'employee_count': 100000,
        'github_org': 'techcorp'
    }


@pytest.fixture
def sample_repository_data():
    """Sample repository data for testing."""
    return {
        'id': 'REPO001',
        'name': 'awesome-project',
        'stars': 5000,
        'forks': 1000,
        'contributors': 50
    }


@pytest.fixture
def sample_sector_data():
    """Sample sector data for testing."""
    return {
        'id': 'SECT001',
        'name': 'Technology',
        'avg_innovation_score': 7.5,
        'avg_digital_maturity': 8.2
    }
