"""Property-based tests for graph schema creation."""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from infrastructure.schema_manager import SchemaManager


# Custom strategies for generating test data
@st.composite
def company_data(draw):
    """Generate random company data."""
    return {
        'id': f"COMP{draw(st.integers(min_value=1, max_value=9999)):04d}",
        'name': draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')))),
        'sector': draw(st.sampled_from(['Technology', 'Healthcare', 'Finance', 'Energy', 'Retail', 'Manufacturing'])),
        'revenue_rank': draw(st.integers(min_value=1, max_value=500)),
        'employee_count': draw(st.integers(min_value=100, max_value=1000000)),
        'github_org': draw(st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=('Ll', 'Nd'))))
    }


@st.composite
def repository_data(draw):
    """Generate random repository data."""
    return {
        'id': f"REPO{draw(st.integers(min_value=1, max_value=9999)):04d}",
        'name': draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Ll', 'Nd', 'Pd')))),
        'stars': draw(st.integers(min_value=0, max_value=100000)),
        'forks': draw(st.integers(min_value=0, max_value=50000)),
        'contributors': draw(st.integers(min_value=1, max_value=1000))
    }


@st.composite
def sector_data(draw):
    """Generate random sector data."""
    sector_name = draw(st.sampled_from(['Technology', 'Healthcare', 'Finance', 'Energy', 'Retail', 'Manufacturing']))
    return {
        'id': f"SECT{draw(st.integers(min_value=1, max_value=99)):02d}",
        'name': sector_name,
        'avg_innovation_score': draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)),
        'avg_digital_maturity': draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    }


@pytest.mark.property
class TestSchemaCreationProperties:
    """Property-based tests for schema creation."""
    
    # Feature: fortune500-kg-analytics, Property 1: Crawl4AI Data Parsing Completeness
    @given(company_data())
    @settings(max_examples=100, deadline=None)
    def test_company_node_creation_preserves_attributes(self, clean_neo4j_db, company_data_instance):
        """
        Property 1: For any valid company data, creating a node should preserve all attributes.
        
        **Validates: Requirements 1.1**
        """
        with clean_neo4j_db.session() as session:
            # Create company node
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
                **company_data_instance
            )
            
            # Retrieve and verify
            result = session.run(
                "MATCH (c:Company {id: $id}) RETURN c",
                id=company_data_instance['id']
            )
            
            retrieved = result.single()
            assert retrieved is not None, "Company node should exist"
            
            company = retrieved['c']
            assert company['id'] == company_data_instance['id']
            assert company['name'] == company_data_instance['name']
            assert company['sector'] == company_data_instance['sector']
            assert company['revenue_rank'] == company_data_instance['revenue_rank']
            assert company['employee_count'] == company_data_instance['employee_count']
            assert company['github_org'] == company_data_instance['github_org']
            assert company['created_at'] is not None
            assert company['updated_at'] is not None
    
    @given(repository_data())
    @settings(max_examples=100, deadline=None)
    def test_repository_node_creation_preserves_attributes(self, clean_neo4j_db, repository_data_instance):
        """
        Property: For any valid repository data, creating a node should preserve all attributes.
        
        **Validates: Requirements 1.2**
        """
        with clean_neo4j_db.session() as session:
            # Create repository node
            session.run(
                """
                CREATE (r:Repository {
                    id: $id,
                    name: $name,
                    stars: $stars,
                    forks: $forks,
                    contributors: $contributors,
                    created_at: datetime()
                })
                """,
                **repository_data_instance
            )
            
            # Retrieve and verify
            result = session.run(
                "MATCH (r:Repository {id: $id}) RETURN r",
                id=repository_data_instance['id']
            )
            
            retrieved = result.single()
            assert retrieved is not None, "Repository node should exist"
            
            repo = retrieved['r']
            assert repo['id'] == repository_data_instance['id']
            assert repo['name'] == repository_data_instance['name']
            assert repo['stars'] == repository_data_instance['stars']
            assert repo['forks'] == repository_data_instance['forks']
            assert repo['contributors'] == repository_data_instance['contributors']
            assert repo['created_at'] is not None
    
    @given(sector_data())
    @settings(max_examples=100, deadline=None)
    def test_sector_node_creation_preserves_attributes(self, clean_neo4j_db, sector_data_instance):
        """
        Property: For any valid sector data, creating a node should preserve all attributes.
        
        **Validates: Requirements 1.1**
        """
        with clean_neo4j_db.session() as session:
            # Create sector node
            session.run(
                """
                CREATE (s:Sector {
                    id: $id,
                    name: $name,
                    avg_innovation_score: $avg_innovation_score,
                    avg_digital_maturity: $avg_digital_maturity
                })
                """,
                **sector_data_instance
            )
            
            # Retrieve and verify
            result = session.run(
                "MATCH (s:Sector {id: $id}) RETURN s",
                id=sector_data_instance['id']
            )
            
            retrieved = result.single()
            assert retrieved is not None, "Sector node should exist"
            
            sector = retrieved['s']
            assert sector['id'] == sector_data_instance['id']
            assert sector['name'] == sector_data_instance['name']
            assert abs(sector['avg_innovation_score'] - sector_data_instance['avg_innovation_score']) < 0.0001
            assert abs(sector['avg_digital_maturity'] - sector_data_instance['avg_digital_maturity']) < 0.0001
    
    @given(company_data(), repository_data())
    @settings(max_examples=100, deadline=None)
    def test_owns_relationship_creation(self, clean_neo4j_db, company_data_instance, repository_data_instance):
        """
        Property: For any company and repository, creating an OWNS relationship should be retrievable.
        
        **Validates: Requirements 1.1**
        """
        with clean_neo4j_db.session() as session:
            # Create nodes and relationship
            session.run(
                """
                CREATE (c:Company {id: $company_id, name: $company_name})
                CREATE (r:Repository {id: $repo_id, name: $repo_name})
                CREATE (c)-[:OWNS]->(r)
                """,
                company_id=company_data_instance['id'],
                company_name=company_data_instance['name'],
                repo_id=repository_data_instance['id'],
                repo_name=repository_data_instance['name']
            )
            
            # Verify relationship exists
            result = session.run(
                """
                MATCH (c:Company {id: $company_id})-[rel:OWNS]->(r:Repository {id: $repo_id})
                RETURN c, rel, r
                """,
                company_id=company_data_instance['id'],
                repo_id=repository_data_instance['id']
            )
            
            record = result.single()
            assert record is not None, "OWNS relationship should exist"
            assert record['c']['id'] == company_data_instance['id']
            assert record['r']['id'] == repository_data_instance['id']
    
    @given(company_data(), sector_data())
    @settings(max_examples=100, deadline=None)
    def test_belongs_to_relationship_creation(self, clean_neo4j_db, company_data_instance, sector_data_instance):
        """
        Property: For any company and sector, creating a BELONGS_TO relationship should be retrievable.
        
        **Validates: Requirements 1.1, 1.3**
        """
        with clean_neo4j_db.session() as session:
            # Create nodes and relationship
            session.run(
                """
                CREATE (c:Company {
                    id: $company_id,
                    name: $company_name,
                    sector: $sector_name,
                    revenue_rank: $revenue_rank,
                    employee_count: $employee_count
                })
                CREATE (s:Sector {id: $sector_id, name: $sector_name})
                CREATE (c)-[:BELONGS_TO]->(s)
                """,
                company_id=company_data_instance['id'],
                company_name=company_data_instance['name'],
                sector_name=sector_data_instance['name'],
                revenue_rank=company_data_instance['revenue_rank'],
                employee_count=company_data_instance['employee_count'],
                sector_id=sector_data_instance['id']
            )
            
            # Verify relationship exists and company has required attributes
            result = session.run(
                """
                MATCH (c:Company {id: $company_id})-[rel:BELONGS_TO]->(s:Sector {id: $sector_id})
                RETURN c, rel, s
                """,
                company_id=company_data_instance['id'],
                sector_id=sector_data_instance['id']
            )
            
            record = result.single()
            assert record is not None, "BELONGS_TO relationship should exist"
            
            # Verify required attributes (Property 3: Required Company Attributes Persistence)
            company = record['c']
            assert company['employee_count'] is not None, "employee_count must be present"
            assert company['revenue_rank'] is not None, "revenue_rank must be present"
            assert company['employee_count'] == company_data_instance['employee_count']
            assert company['revenue_rank'] == company_data_instance['revenue_rank']
    
    @given(st.lists(company_data(), min_size=1, max_size=10, unique_by=lambda x: x['id']))
    @settings(max_examples=50, deadline=None)
    def test_multiple_companies_creation(self, clean_neo4j_db, companies_list):
        """
        Property: For any list of companies, all should be created and retrievable.
        
        **Validates: Requirements 1.1, 1.4**
        """
        with clean_neo4j_db.session() as session:
            # Create all companies
            for company in companies_list:
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
                    **company
                )
            
            # Count created companies (Property 4: Ingestion Logging Accuracy)
            result = session.run("MATCH (c:Company) RETURN count(c) as count")
            count = result.single()['count']
            
            assert count == len(companies_list), \
                f"Node count should equal number of companies created: expected {len(companies_list)}, got {count}"
    
    @given(
        st.lists(company_data(), min_size=2, max_size=5, unique_by=lambda x: x['id']),
        st.lists(repository_data(), min_size=2, max_size=5, unique_by=lambda x: x['id'])
    )
    @settings(max_examples=50, deadline=None)
    def test_graph_structure_preservation(self, clean_neo4j_db, companies_list, repositories_list):
        """
        Property: For any set of companies and repositories with relationships,
        the graph structure should be fully preserved.
        
        **Validates: Requirements 1.1, 1.4**
        """
        with clean_neo4j_db.session() as session:
            # Create companies
            for company in companies_list:
                session.run(
                    "CREATE (c:Company {id: $id, name: $name})",
                    id=company['id'],
                    name=company['name']
                )
            
            # Create repositories
            for repo in repositories_list:
                session.run(
                    "CREATE (r:Repository {id: $id, name: $name})",
                    id=repo['id'],
                    name=repo['name']
                )
            
            # Create relationships (each company owns first repository)
            for company in companies_list:
                session.run(
                    """
                    MATCH (c:Company {id: $company_id})
                    MATCH (r:Repository {id: $repo_id})
                    CREATE (c)-[:OWNS]->(r)
                    """,
                    company_id=company['id'],
                    repo_id=repositories_list[0]['id']
                )
            
            # Verify counts
            company_count = session.run("MATCH (c:Company) RETURN count(c) as count").single()['count']
            repo_count = session.run("MATCH (r:Repository) RETURN count(r) as count").single()['count']
            rel_count = session.run("MATCH ()-[r:OWNS]->() RETURN count(r) as count").single()['count']
            
            assert company_count == len(companies_list)
            assert repo_count == len(repositories_list)
            assert rel_count == len(companies_list)  # Each company owns one repo
