"""Property-based tests for Crawl4AI data parsing.

This module contains property-based tests using the hypothesis library
to verify that the DataIngestionPipeline correctly parses and stores
Crawl4AI data to the Knowledge Graph.
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from datetime import datetime

from fortune500_kg.data_ingestion_pipeline import DataIngestionPipeline
from fortune500_kg.data_models import CrawlData


# Custom strategies for generating test data
@st.composite
def company_data_strategy(draw):
    """Generate valid company data for testing."""
    # Generate unique company ID to avoid MERGE conflicts
    company_id = draw(st.text(min_size=5, max_size=20, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'), min_codepoint=48, max_codepoint=122
    )))
    
    return {
        'id': f"COMP{company_id}",
        'name': draw(st.text(min_size=1, max_size=100)),
        'sector': draw(st.sampled_from([
            'Technology', 'Finance', 'Healthcare', 'Retail', 
            'Manufacturing', 'Energy', 'Telecommunications'
        ])),
        'revenue_rank': draw(st.integers(min_value=1, max_value=500)),
        'employee_count': draw(st.integers(min_value=1, max_value=1000000)),
        'github_org': draw(st.one_of(
            st.none(),
            st.text(min_size=1, max_size=50, alphabet=st.characters(
                whitelist_categories=('Ll', 'Nd'), min_codepoint=48, max_codepoint=122
            ))
        ))
    }


@st.composite
def relationship_data_strategy(draw, company_ids):
    """Generate valid relationship data for testing."""
    if len(company_ids) < 2:
        return None
    
    # Get unique company IDs
    unique_ids = list(set(company_ids))
    if len(unique_ids) < 2:
        return None
    
    from_id = draw(st.sampled_from(unique_ids))
    to_id = draw(st.sampled_from([cid for cid in unique_ids if cid != from_id]))
    
    return {
        'from_id': from_id,
        'to_id': to_id,
        'relationship_type': draw(st.sampled_from([
            'PARTNERS_WITH', 'ACQUIRED', 'DEPENDS_ON'
        ])),
        'properties': draw(st.dictionaries(
            keys=st.sampled_from(['since', 'partnership_type', 'amount']),
            values=st.one_of(
                st.text(min_size=1, max_size=50),
                st.integers(min_value=0, max_value=1000000),
                st.floats(min_value=0, max_value=1000000, allow_nan=False, allow_infinity=False)
            ),
            max_size=3
        ))
    }


@st.composite
def crawl_data_strategy(draw):
    """Generate valid CrawlData structures for testing."""
    # Generate 1-10 companies
    num_companies = draw(st.integers(min_value=1, max_value=10))
    companies = [draw(company_data_strategy()) for _ in range(num_companies)]
    
    # Extract company IDs
    company_ids = [c['id'] for c in companies]
    
    # Generate 0-5 relationships
    num_relationships = draw(st.integers(min_value=0, max_value=min(5, len(company_ids))))
    relationships = []
    
    if len(company_ids) >= 2:
        for _ in range(num_relationships):
            rel = draw(relationship_data_strategy(company_ids))
            if rel:
                relationships.append(rel)
    
    return CrawlData(
        companies=companies,
        relationships=relationships,
        metadata={'source': 'test', 'timestamp': datetime.now().isoformat()}
    )


class TestCrawl4AIParsingProperties:
    """Property-based tests for Crawl4AI data parsing completeness."""
    
    @given(crawl_data=crawl_data_strategy())
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_1_crawl4ai_data_parsing_completeness(
        self, 
        crawl_data, 
        clean_neo4j_db
    ):
        """
        Property 1: Crawl4AI Data Parsing Completeness
        
        For any valid Crawl4AI data structure containing company information,
        parsing and storing to the Knowledge Graph should result in nodes and
        relationships that preserve all company attributes and connections from
        the source data.
        
        **Validates: Requirements 1.1**
        
        This property verifies that:
        1. All companies in the source data are created as nodes
        2. All company attributes are preserved
        3. All relationships are created as edges
        4. No data is lost during parsing
        """
        # Arrange
        pipeline = DataIngestionPipeline(clean_neo4j_db)
        
        # Act
        result = pipeline.ingest_crawl4ai_data(crawl_data)
        
        # Calculate expected counts accounting for duplicate IDs (MERGE behavior)
        unique_company_ids = set(c['id'] for c in crawl_data.companies)
        expected_node_count = len(unique_company_ids)
        
        # For relationships, count unique (from_id, to_id, type) tuples
        unique_relationships = set(
            (r['from_id'], r['to_id'], r['relationship_type'])
            for r in crawl_data.relationships
        )
        expected_edge_count = len(unique_relationships)
        
        # Assert: All unique companies should be ingested
        assert result.node_count == expected_node_count, (
            f"Expected {expected_node_count} nodes, but got {result.node_count}"
        )
        
        # Assert: All unique relationships should be created
        assert result.edge_count == expected_edge_count, (
            f"Expected {expected_edge_count} edges, but got {result.edge_count}. "
            f"Errors: {result.errors}"
        )
        
        # Assert: No errors should occur with valid data
        assert len(result.errors) == 0, (
            f"Expected no errors, but got: {result.errors}"
        )
        
        # Verify data preservation in the Knowledge Graph
        # For duplicate IDs, the last company data should be preserved (MERGE behavior)
        company_by_id = {}
        for company in crawl_data.companies:
            company_by_id[company['id']] = company
        
        with clean_neo4j_db.session() as session:
            # Verify all unique companies exist with correct attributes
            for company_id, company in company_by_id.items():
                query = """
                MATCH (c:Company {id: $id})
                RETURN c.id as id, 
                       c.name as name, 
                       c.sector as sector,
                       c.revenue_rank as revenue_rank,
                       c.employee_count as employee_count,
                       c.github_org as github_org
                """
                result_record = session.run(query, id=company_id).single()
                
                assert result_record is not None, (
                    f"Company {company_id} not found in Knowledge Graph"
                )
                
                # Verify all attributes are preserved
                assert result_record['id'] == company['id']
                assert result_record['name'] == company['name']
                assert result_record['sector'] == company['sector']
                assert result_record['revenue_rank'] == company['revenue_rank']
                assert result_record['employee_count'] == company['employee_count']
                assert result_record['github_org'] == company.get('github_org')
            
            # Verify all unique relationships exist
            for relationship in crawl_data.relationships:
                query = f"""
                MATCH (from {{id: $from_id}})-[r:{relationship['relationship_type']}]->(to {{id: $to_id}})
                RETURN r
                """
                result_record = session.run(
                    query,
                    from_id=relationship['from_id'],
                    to_id=relationship['to_id']
                ).single()
                
                assert result_record is not None, (
                    f"Relationship {relationship['relationship_type']} from "
                    f"{relationship['from_id']} to {relationship['to_id']} "
                    f"not found in Knowledge Graph"
                )
    
    @given(
        companies=st.lists(
            company_data_strategy(),
            min_size=1,
            max_size=20
        )
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_1_attribute_preservation(
        self,
        companies,
        clean_neo4j_db
    ):
        """
        Property 1 (Extended): Attribute Preservation
        
        For any set of companies with various attribute combinations,
        all attributes should be preserved exactly in the Knowledge Graph.
        
        **Validates: Requirements 1.1, 1.3**
        """
        # Arrange
        pipeline = DataIngestionPipeline(clean_neo4j_db)
        crawl_data = CrawlData(companies=companies, relationships=[])
        
        # Act
        result = pipeline.ingest_crawl4ai_data(crawl_data)
        
        # Calculate expected unique count
        unique_company_ids = set(c['id'] for c in companies)
        expected_count = len(unique_company_ids)
        
        # Assert: All unique companies ingested successfully
        assert result.node_count == expected_count
        assert len(result.errors) == 0
        
        # Build map of last company data per ID (MERGE behavior)
        company_by_id = {}
        for company in companies:
            company_by_id[company['id']] = company
        
        # Verify attribute preservation
        with clean_neo4j_db.session() as session:
            for company_id, company in company_by_id.items():
                query = """
                MATCH (c:Company {id: $id})
                RETURN c
                """
                result_record = session.run(query, id=company_id).single()
                
                assert result_record is not None
                node = result_record['c']
                
                # Verify required attributes
                assert node['id'] == company['id']
                assert node['name'] == company['name']
                assert node['sector'] == company['sector']
                assert node['revenue_rank'] == company['revenue_rank']
                assert node['employee_count'] == company['employee_count']
                
                # Verify optional github_org attribute
                if company.get('github_org'):
                    assert node['github_org'] == company['github_org']
    
    @given(crawl_data=crawl_data_strategy())
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_1_relationship_preservation(
        self,
        crawl_data,
        clean_neo4j_db
    ):
        """
        Property 1 (Extended): Relationship Preservation
        
        For any valid CrawlData with relationships, all relationship types
        and properties should be preserved in the Knowledge Graph.
        
        **Validates: Requirements 1.1**
        """
        # Arrange
        pipeline = DataIngestionPipeline(clean_neo4j_db)
        
        # Act
        result = pipeline.ingest_crawl4ai_data(crawl_data)
        
        # Calculate expected unique relationship count
        unique_relationships = set(
            (r['from_id'], r['to_id'], r['relationship_type'])
            for r in crawl_data.relationships
        )
        expected_edge_count = len(unique_relationships)
        
        # Assert: All unique relationships created
        assert result.edge_count == expected_edge_count
        
        # Verify relationship preservation (check unique relationships)
        with clean_neo4j_db.session() as session:
            for from_id, to_id, rel_type in unique_relationships:
                query = f"""
                MATCH (from {{id: $from_id}})-[r:{rel_type}]->(to {{id: $to_id}})
                RETURN r, type(r) as rel_type
                """
                result_record = session.run(
                    query,
                    from_id=from_id,
                    to_id=to_id
                ).single()
                
                assert result_record is not None
                assert result_record['rel_type'] == rel_type
    
    @given(
        num_companies=st.integers(min_value=0, max_value=20)
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_1_count_accuracy(
        self,
        num_companies,
        clean_neo4j_db
    ):
        """
        Property 1 (Extended): Count Accuracy
        
        For any number of companies N, the ingestion result should report
        exactly N nodes created.
        
        **Validates: Requirements 1.1, 1.4**
        """
        # Arrange
        pipeline = DataIngestionPipeline(clean_neo4j_db)
        
        # Generate exactly num_companies companies with unique IDs
        companies = []
        for i in range(num_companies):
            companies.append({
                'id': f'COMPTEST{i:06d}',  # Unique prefix to avoid conflicts
                'name': f'Company {i}',
                'sector': 'Technology',
                'revenue_rank': i + 1,
                'employee_count': (i + 1) * 1000
            })
        
        crawl_data = CrawlData(companies=companies, relationships=[])
        
        # Act
        result = pipeline.ingest_crawl4ai_data(crawl_data)
        
        # Assert: Reported count matches actual count
        assert result.node_count == num_companies
        
        # Verify actual count in database
        with clean_neo4j_db.session() as session:
            query = "MATCH (c:Company) WHERE c.id STARTS WITH 'COMPTEST' RETURN count(c) as count"
            db_count = session.run(query).single()['count']
            assert db_count == num_companies
    
    @given(crawl_data=crawl_data_strategy())
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_1_idempotency(
        self,
        crawl_data,
        clean_neo4j_db
    ):
        """
        Property 1 (Extended): Idempotency
        
        For any valid CrawlData, ingesting the same data twice should
        result in the same final state (MERGE behavior).
        
        **Validates: Requirements 1.1**
        """
        # Arrange
        pipeline = DataIngestionPipeline(clean_neo4j_db)
        
        # Act: Ingest twice
        result1 = pipeline.ingest_crawl4ai_data(crawl_data)
        result2 = pipeline.ingest_crawl4ai_data(crawl_data)
        
        # Assert: Both ingestions report same counts
        assert result1.node_count == result2.node_count
        assert result1.edge_count == result2.edge_count
        
        # Calculate expected unique counts
        unique_company_ids = set(c['id'] for c in crawl_data.companies)
        unique_relationships = set(
            (r['from_id'], r['to_id'], r['relationship_type'])
            for r in crawl_data.relationships
        )
        
        # Verify database state - count only the companies from this test
        with clean_neo4j_db.session() as session:
            # Count companies with IDs from this test data
            company_ids = list(unique_company_ids)
            if company_ids:
                query = "MATCH (c:Company) WHERE c.id IN $ids RETURN count(c) as count"
                db_count = session.run(query, ids=company_ids).single()['count']
                assert db_count == len(unique_company_ids)
            
            # Count relationships involving these companies
            if company_ids and unique_relationships:
                query = """
                MATCH (from)-[r]->(to)
                WHERE from.id IN $ids AND to.id IN $ids
                RETURN count(r) as count
                """
                rel_count = session.run(query, ids=company_ids).single()['count']
                assert rel_count == len(unique_relationships)
