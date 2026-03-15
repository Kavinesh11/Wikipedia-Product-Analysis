"""Unit tests for Neo4j infrastructure."""

import pytest
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from infrastructure.connection import Neo4jConnection
from infrastructure.schema_manager import SchemaManager


class TestNeo4jConnection:
    """Test Neo4j connection management."""
    
    def test_connection_initialization(self):
        """Test connection object initialization."""
        conn = Neo4jConnection(
            uri='bolt://localhost:7688',
            user='neo4j',
            password='testpassword'
        )
        assert conn.uri == 'bolt://localhost:7688'
        assert conn.user == 'neo4j'
        assert conn.password == 'testpassword'
    
    def test_connection_context_manager(self, neo4j_test_uri, neo4j_test_credentials):
        """Test connection as context manager."""
        user, password = neo4j_test_credentials
        with Neo4jConnection(neo4j_test_uri, user, password) as conn:
            assert conn._driver is not None
        
        # Driver should be closed after context exit
        assert conn._driver is None
    
    def test_verify_connectivity(self, neo4j_test_uri, neo4j_test_credentials):
        """Test connection verification."""
        user, password = neo4j_test_credentials
        conn = Neo4jConnection(neo4j_test_uri, user, password)
        
        result = conn.verify_connectivity()
        assert result is True
        
        conn.close()


class TestSchemaManager:
    """Test schema management functionality."""
    
    def test_schema_manager_initialization(self, neo4j_test_driver):
        """Test schema manager initialization."""
        manager = SchemaManager(neo4j_test_driver)
        assert manager.driver is not None
        assert manager.schema_dir.exists()
    
    def test_create_constraints(self, clean_neo4j_db):
        """Test constraint creation."""
        manager = SchemaManager(clean_neo4j_db)
        manager.create_constraints()
        
        constraints = manager.get_constraints()
        assert len(constraints) > 0
        
        # Check for specific constraints
        constraint_names = [c.get('name', '').lower() for c in constraints]
        assert any('company' in name for name in constraint_names)
    
    def test_create_indexes(self, clean_neo4j_db):
        """Test index creation."""
        manager = SchemaManager(clean_neo4j_db)
        manager.create_constraints()  # Constraints must exist first
        manager.create_indexes()
        
        indexes = manager.get_indexes()
        assert len(indexes) > 0
        
        # Check for specific indexes
        index_names = [idx.get('name', '').lower() for idx in indexes]
        assert any('sector' in name or 'revenue' in name for name in index_names)
    
    def test_initialize_schema(self, clean_neo4j_db):
        """Test complete schema initialization."""
        manager = SchemaManager(clean_neo4j_db)
        manager.initialize_schema()
        
        # Verify constraints and indexes were created
        constraints = manager.get_constraints()
        indexes = manager.get_indexes()
        
        assert len(constraints) > 0
        assert len(indexes) > 0
    
    def test_validate_schema(self, clean_neo4j_db):
        """Test schema validation."""
        manager = SchemaManager(clean_neo4j_db)
        manager.initialize_schema()
        
        validation = manager.validate_schema()
        
        assert validation['constraints_exist'] is True
        assert validation['indexes_exist'] is True
    
    def test_database_stats_empty(self, clean_neo4j_db):
        """Test database statistics on empty database."""
        manager = SchemaManager(clean_neo4j_db)
        stats = manager.get_database_stats()
        
        assert stats['companies'] == 0
        assert stats['repositories'] == 0
        assert stats['sectors'] == 0
        assert stats['owns_relationships'] == 0
    
    def test_database_stats_with_data(self, clean_neo4j_db, sample_company_data):
        """Test database statistics with sample data."""
        manager = SchemaManager(clean_neo4j_db)
        
        # Create a sample company node
        with clean_neo4j_db.session() as session:
            session.run(
                """
                CREATE (c:Company {
                    id: $id,
                    name: $name,
                    sector: $sector,
                    revenue_rank: $revenue_rank,
                    employee_count: $employee_count,
                    github_org: $github_org,
                    created_at: datetime(),
                    updated_at: datetime()
                })
                """,
                **sample_company_data
            )
        
        stats = manager.get_database_stats()
        assert stats['companies'] == 1
        assert stats['repositories'] == 0
    
    def test_clear_database(self, clean_neo4j_db, sample_company_data):
        """Test database clearing."""
        manager = SchemaManager(clean_neo4j_db)
        
        # Create sample data
        with clean_neo4j_db.session() as session:
            session.run(
                "CREATE (c:Company {id: $id, name: $name})",
                id=sample_company_data['id'],
                name=sample_company_data['name']
            )
        
        # Verify data exists
        stats_before = manager.get_database_stats()
        assert stats_before['companies'] == 1
        
        # Clear database
        manager.clear_database()
        
        # Verify data is gone
        stats_after = manager.get_database_stats()
        assert stats_after['companies'] == 0


class TestGraphSchema:
    """Test graph schema structure."""
    
    def test_create_company_node(self, clean_neo4j_db, sample_company_data):
        """Test creating a Company node with all required properties."""
        with clean_neo4j_db.session() as session:
            result = session.run(
                """
                CREATE (c:Company {
                    id: $id,
                    name: $name,
                    sector: $sector,
                    revenue_rank: $revenue_rank,
                    employee_count: $employee_count,
                    github_org: $github_org,
                    created_at: datetime(),
                    updated_at: datetime()
                })
                RETURN c
                """,
                **sample_company_data
            )
            
            company = result.single()['c']
            assert company['id'] == sample_company_data['id']
            assert company['name'] == sample_company_data['name']
            assert company['sector'] == sample_company_data['sector']
    
    def test_create_repository_node(self, clean_neo4j_db, sample_repository_data):
        """Test creating a Repository node with all required properties."""
        with clean_neo4j_db.session() as session:
            result = session.run(
                """
                CREATE (r:Repository {
                    id: $id,
                    name: $name,
                    stars: $stars,
                    forks: $forks,
                    contributors: $contributors,
                    created_at: datetime()
                })
                RETURN r
                """,
                **sample_repository_data
            )
            
            repo = result.single()['r']
            assert repo['id'] == sample_repository_data['id']
            assert repo['stars'] == sample_repository_data['stars']
    
    def test_create_sector_node(self, clean_neo4j_db, sample_sector_data):
        """Test creating a Sector node with all required properties."""
        with clean_neo4j_db.session() as session:
            result = session.run(
                """
                CREATE (s:Sector {
                    id: $id,
                    name: $name,
                    avg_innovation_score: $avg_innovation_score,
                    avg_digital_maturity: $avg_digital_maturity
                })
                RETURN s
                """,
                **sample_sector_data
            )
            
            sector = result.single()['s']
            assert sector['id'] == sample_sector_data['id']
            assert sector['name'] == sample_sector_data['name']
    
    def test_create_owns_relationship(
        self, clean_neo4j_db, sample_company_data, sample_repository_data
    ):
        """Test creating OWNS relationship between Company and Repository."""
        with clean_neo4j_db.session() as session:
            session.run(
                """
                CREATE (c:Company {id: $company_id, name: $company_name})
                CREATE (r:Repository {id: $repo_id, name: $repo_name})
                CREATE (c)-[:OWNS]->(r)
                """,
                company_id=sample_company_data['id'],
                company_name=sample_company_data['name'],
                repo_id=sample_repository_data['id'],
                repo_name=sample_repository_data['name']
            )
            
            result = session.run(
                "MATCH (c:Company)-[rel:OWNS]->(r:Repository) RETURN count(rel) as count"
            )
            assert result.single()['count'] == 1
    
    def test_create_belongs_to_relationship(
        self, clean_neo4j_db, sample_company_data, sample_sector_data
    ):
        """Test creating BELONGS_TO relationship between Company and Sector."""
        with clean_neo4j_db.session() as session:
            session.run(
                """
                CREATE (c:Company {id: $company_id, name: $company_name})
                CREATE (s:Sector {id: $sector_id, name: $sector_name})
                CREATE (c)-[:BELONGS_TO]->(s)
                """,
                company_id=sample_company_data['id'],
                company_name=sample_company_data['name'],
                sector_id=sample_sector_data['id'],
                sector_name=sample_sector_data['name']
            )
            
            result = session.run(
                "MATCH (c:Company)-[rel:BELONGS_TO]->(s:Sector) RETURN count(rel) as count"
            )
            assert result.single()['count'] == 1
    
    def test_create_partners_with_relationship(self, clean_neo4j_db):
        """Test creating PARTNERS_WITH relationship with properties."""
        with clean_neo4j_db.session() as session:
            session.run(
                """
                CREATE (c1:Company {id: 'COMP001', name: 'Company A'})
                CREATE (c2:Company {id: 'COMP002', name: 'Company B'})
                CREATE (c1)-[:PARTNERS_WITH {
                    since: date('2023-01-01'),
                    partnership_type: 'Strategic Alliance'
                }]->(c2)
                """
            )
            
            result = session.run(
                """
                MATCH (c1:Company)-[rel:PARTNERS_WITH]->(c2:Company)
                RETURN rel.partnership_type as type
                """
            )
            assert result.single()['type'] == 'Strategic Alliance'
    
    def test_create_acquired_relationship(self, clean_neo4j_db):
        """Test creating ACQUIRED relationship with properties."""
        with clean_neo4j_db.session() as session:
            session.run(
                """
                CREATE (c1:Company {id: 'COMP001', name: 'Acquirer'})
                CREATE (c2:Company {id: 'COMP002', name: 'Target'})
                CREATE (c1)-[:ACQUIRED {
                    date: date('2023-06-15'),
                    amount: 1000000000.0
                }]->(c2)
                """
            )
            
            result = session.run(
                """
                MATCH (c1:Company)-[rel:ACQUIRED]->(c2:Company)
                RETURN rel.amount as amount
                """
            )
            assert result.single()['amount'] == 1000000000.0
    
    def test_create_depends_on_relationship(
        self, clean_neo4j_db, sample_company_data, sample_repository_data
    ):
        """Test creating DEPENDS_ON relationship with properties."""
        with clean_neo4j_db.session() as session:
            session.run(
                """
                CREATE (c:Company {id: $company_id, name: $company_name})
                CREATE (r:Repository {id: $repo_id, name: $repo_name})
                CREATE (c)-[:DEPENDS_ON {dependency_type: 'direct'}]->(r)
                """,
                company_id=sample_company_data['id'],
                company_name=sample_company_data['name'],
                repo_id=sample_repository_data['id'],
                repo_name=sample_repository_data['name']
            )
            
            result = session.run(
                """
                MATCH (c:Company)-[rel:DEPENDS_ON]->(r:Repository)
                RETURN rel.dependency_type as type
                """
            )
            assert result.single()['type'] == 'direct'
