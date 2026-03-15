"""Unit tests for DataIngestionPipeline."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from fortune500_kg.data_ingestion_pipeline import DataIngestionPipeline
from fortune500_kg.data_models import CrawlData


@pytest.fixture
def mock_driver():
    """Create a mock Neo4j driver."""
    driver = Mock()
    session = MagicMock()
    
    # Mock execute_write to return True for successful operations
    session.execute_write.return_value = True
    
    # Properly mock the context manager
    mock_session_context = MagicMock()
    mock_session_context.__enter__ = Mock(return_value=session)
    mock_session_context.__exit__ = Mock(return_value=None)
    driver.session.return_value = mock_session_context
    
    return driver


class TestDataIngestionPipeline:
    """Unit tests for DataIngestionPipeline class."""
    
    def test_ingest_single_company(self, mock_driver):
        """Test ingesting a single company with all required attributes."""
        pipeline = DataIngestionPipeline(mock_driver)
        
        crawl_data = CrawlData(
            companies=[{
                'id': 'COMP001',
                'name': 'Tech Corp',
                'sector': 'Technology',
                'revenue_rank': 1,
                'employee_count': 100000,
                'github_org': 'techcorp'
            }],
            relationships=[]
        )
        
        result = pipeline.ingest_crawl4ai_data(crawl_data)
        
        assert result.node_count == 1
        assert result.edge_count == 0
        assert len(result.errors) == 0
    
    def test_ingest_multiple_companies(self, mock_driver):
        """Test ingesting multiple companies."""
        pipeline = DataIngestionPipeline(mock_driver)
        
        crawl_data = CrawlData(
            companies=[
                {
                    'id': 'COMP001',
                    'name': 'Tech Corp',
                    'sector': 'Technology',
                    'revenue_rank': 1,
                    'employee_count': 100000,
                    'github_org': 'techcorp'
                },
                {
                    'id': 'COMP002',
                    'name': 'Finance Inc',
                    'sector': 'Finance',
                    'revenue_rank': 2,
                    'employee_count': 50000,
                    'github_org': 'financeinc'
                },
                {
                    'id': 'COMP003',
                    'name': 'Retail Co',
                    'sector': 'Retail',
                    'revenue_rank': 3,
                    'employee_count': 75000
                }
            ],
            relationships=[]
        )
        
        result = pipeline.ingest_crawl4ai_data(crawl_data)
        
        assert result.node_count == 3
        assert result.edge_count == 0
        assert len(result.errors) == 0
    
    def test_ingest_company_with_relationship(self, mock_driver):
        """Test ingesting companies with relationships."""
        pipeline = DataIngestionPipeline(mock_driver)
        
        crawl_data = CrawlData(
            companies=[
                {
                    'id': 'COMP001',
                    'name': 'Tech Corp',
                    'sector': 'Technology',
                    'revenue_rank': 1,
                    'employee_count': 100000
                },
                {
                    'id': 'COMP002',
                    'name': 'Finance Inc',
                    'sector': 'Finance',
                    'revenue_rank': 2,
                    'employee_count': 50000
                }
            ],
            relationships=[
                {
                    'from_id': 'COMP001',
                    'to_id': 'COMP002',
                    'relationship_type': 'PARTNERS_WITH',
                    'properties': {
                        'since': '2020-01-01',
                        'partnership_type': 'strategic'
                    }
                }
            ]
        )
        
        result = pipeline.ingest_crawl4ai_data(crawl_data)
        
        assert result.node_count == 2
        assert result.edge_count == 1
        assert len(result.errors) == 0
    
    def test_ingest_missing_required_field(self, mock_driver):
        """Test ingesting company with missing required field."""
        pipeline = DataIngestionPipeline(mock_driver)
        
        crawl_data = CrawlData(
            companies=[
                {
                    'id': 'COMP001',
                    'name': 'Tech Corp',
                    'sector': 'Technology',
                    # Missing revenue_rank
                    'employee_count': 100000
                }
            ],
            relationships=[]
        )
        
        result = pipeline.ingest_crawl4ai_data(crawl_data)
        
        assert result.node_count == 0
        assert len(result.errors) > 0
        assert 'Invalid company data' in result.errors[0]
    
    def test_ingest_invalid_data_types(self, mock_driver):
        """Test ingesting company with invalid data types."""
        pipeline = DataIngestionPipeline(mock_driver)
        
        crawl_data = CrawlData(
            companies=[
                {
                    'id': 'COMP001',
                    'name': 'Tech Corp',
                    'sector': 'Technology',
                    'revenue_rank': 'one',  # Should be int
                    'employee_count': 100000
                }
            ],
            relationships=[]
        )
        
        result = pipeline.ingest_crawl4ai_data(crawl_data)
        
        assert result.node_count == 0
        assert len(result.errors) > 0
    
    def test_ingest_empty_dataset(self, mock_driver):
        """Test ingesting empty dataset."""
        pipeline = DataIngestionPipeline(mock_driver)
        
        crawl_data = CrawlData(
            companies=[],
            relationships=[]
        )
        
        result = pipeline.ingest_crawl4ai_data(crawl_data)
        
        assert result.node_count == 0
        assert result.edge_count == 0
        assert len(result.errors) == 0
    
    def test_ingest_duplicate_company(self, mock_driver):
        """Test ingesting the same company twice (should update via MERGE)."""
        pipeline = DataIngestionPipeline(mock_driver)
        
        # First ingestion
        crawl_data1 = CrawlData(
            companies=[{
                'id': 'COMP001',
                'name': 'Tech Corp',
                'sector': 'Technology',
                'revenue_rank': 1,
                'employee_count': 100000
            }],
            relationships=[]
        )
        
        result1 = pipeline.ingest_crawl4ai_data(crawl_data1)
        assert result1.node_count == 1
        
        # Second ingestion with updated data
        crawl_data2 = CrawlData(
            companies=[{
                'id': 'COMP001',
                'name': 'Tech Corp Updated',
                'sector': 'Technology',
                'revenue_rank': 1,
                'employee_count': 120000
            }],
            relationships=[]
        )
        
        result2 = pipeline.ingest_crawl4ai_data(crawl_data2)
        assert result2.node_count == 1
    
    def test_validate_company_data_valid(self, mock_driver):
        """Test validation of valid company data."""
        pipeline = DataIngestionPipeline(mock_driver)
        
        valid_data = {
            'id': 'COMP001',
            'name': 'Tech Corp',
            'sector': 'Technology',
            'revenue_rank': 1,
            'employee_count': 100000
        }
        
        assert pipeline._validate_company_data(valid_data) is True
    
    def test_validate_company_data_missing_field(self, mock_driver):
        """Test validation of company data with missing field."""
        pipeline = DataIngestionPipeline(mock_driver)
        
        invalid_data = {
            'id': 'COMP001',
            'name': 'Tech Corp',
            'sector': 'Technology',
            # Missing revenue_rank and employee_count
        }
        
        assert pipeline._validate_company_data(invalid_data) is False
    
    def test_validate_company_data_none_values(self, mock_driver):
        """Test validation of company data with None values."""
        pipeline = DataIngestionPipeline(mock_driver)
        
        invalid_data = {
            'id': 'COMP001',
            'name': 'Tech Corp',
            'sector': 'Technology',
            'revenue_rank': None,
            'employee_count': 100000
        }
        
        assert pipeline._validate_company_data(invalid_data) is False
    
    def test_validate_relationship_data_valid(self, mock_driver):
        """Test validation of valid relationship data."""
        pipeline = DataIngestionPipeline(mock_driver)
        
        valid_data = {
            'from_id': 'COMP001',
            'to_id': 'COMP002',
            'relationship_type': 'PARTNERS_WITH'
        }
        
        assert pipeline._validate_relationship_data(valid_data) is True
    
    def test_validate_relationship_data_missing_field(self, mock_driver):
        """Test validation of relationship data with missing field."""
        pipeline = DataIngestionPipeline(mock_driver)
        
        invalid_data = {
            'from_id': 'COMP001',
            # Missing to_id and relationship_type
        }
        
        assert pipeline._validate_relationship_data(invalid_data) is False
    
    def test_ingestion_result_timestamp(self, mock_driver):
        """Test that ingestion result includes timestamp."""
        pipeline = DataIngestionPipeline(mock_driver)
        
        crawl_data = CrawlData(companies=[], relationships=[])
        result = pipeline.ingest_crawl4ai_data(crawl_data)
        
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)
    
    def test_ingest_company_without_github_org(self, mock_driver):
        """Test ingesting company without github_org (optional field)."""
        pipeline = DataIngestionPipeline(mock_driver)
        
        crawl_data = CrawlData(
            companies=[{
                'id': 'COMP001',
                'name': 'Tech Corp',
                'sector': 'Technology',
                'revenue_rank': 1,
                'employee_count': 100000
                # No github_org
            }],
            relationships=[]
        )
        
        result = pipeline.ingest_crawl4ai_data(crawl_data)
        
        assert result.node_count == 1
        assert len(result.errors) == 0
    
    def test_ingest_relationship_invalid_data(self, mock_driver):
        """Test ingesting relationship with invalid data."""
        pipeline = DataIngestionPipeline(mock_driver)
        
        crawl_data = CrawlData(
            companies=[
                {
                    'id': 'COMP001',
                    'name': 'Tech Corp',
                    'sector': 'Technology',
                    'revenue_rank': 1,
                    'employee_count': 100000
                }
            ],
            relationships=[
                {
                    'from_id': 'COMP001',
                    # Missing to_id and relationship_type
                }
            ]
        )
        
        result = pipeline.ingest_crawl4ai_data(crawl_data)
        
        assert result.node_count == 1
        assert result.edge_count == 0
        assert len(result.errors) > 0
        assert 'Invalid relationship data' in result.errors[0]
    
    def test_ingest_multiple_relationships(self, mock_driver):
        """Test ingesting multiple relationships."""
        pipeline = DataIngestionPipeline(mock_driver)
        
        crawl_data = CrawlData(
            companies=[
                {
                    'id': 'COMP001',
                    'name': 'Tech Corp',
                    'sector': 'Technology',
                    'revenue_rank': 1,
                    'employee_count': 100000
                },
                {
                    'id': 'COMP002',
                    'name': 'Finance Inc',
                    'sector': 'Finance',
                    'revenue_rank': 2,
                    'employee_count': 50000
                },
                {
                    'id': 'COMP003',
                    'name': 'Retail Co',
                    'sector': 'Retail',
                    'revenue_rank': 3,
                    'employee_count': 75000
                }
            ],
            relationships=[
                {
                    'from_id': 'COMP001',
                    'to_id': 'COMP002',
                    'relationship_type': 'PARTNERS_WITH'
                },
                {
                    'from_id': 'COMP001',
                    'to_id': 'COMP003',
                    'relationship_type': 'ACQUIRED'
                }
            ]
        )
        
        result = pipeline.ingest_crawl4ai_data(crawl_data)
        
        assert result.node_count == 3
        assert result.edge_count == 2
        assert len(result.errors) == 0
    
    def test_ingest_handles_database_errors(self):
        """Test that pipeline handles database errors gracefully."""
        # Create a fresh mock driver for this test
        driver = Mock()
        session = MagicMock()
        session.execute_write.side_effect = Exception("Database error")
        
        # Properly mock the context manager
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = Mock(return_value=session)
        mock_session_context.__exit__ = Mock(return_value=None)
        driver.session.return_value = mock_session_context
        
        pipeline = DataIngestionPipeline(driver)
        
        crawl_data = CrawlData(
            companies=[{
                'id': 'COMP001',
                'name': 'Tech Corp',
                'sector': 'Technology',
                'revenue_rank': 1,
                'employee_count': 100000
            }],
            relationships=[]
        )
        
        result = pipeline.ingest_crawl4ai_data(crawl_data)
        
        assert result.node_count == 0
        assert len(result.errors) > 0
        assert 'Error processing company' in result.errors[0]


class TestIngestionLogging:
    """Tests for ingestion logging (Requirement 1.4)."""

    def test_ingest_logs_node_and_edge_counts(self, mock_driver, caplog):
        """Verify that node and edge counts are logged after ingestion."""
        import logging
        pipeline = DataIngestionPipeline(mock_driver)

        crawl_data = CrawlData(
            companies=[
                {
                    'id': 'COMP001',
                    'name': 'Alpha Corp',
                    'sector': 'Technology',
                    'revenue_rank': 1,
                    'employee_count': 50000,
                    'github_org': 'alphacorp',
                },
                {
                    'id': 'COMP002',
                    'name': 'Beta Inc',
                    'sector': 'Finance',
                    'revenue_rank': 2,
                    'employee_count': 30000,
                },
            ],
            relationships=[
                {
                    'from_id': 'COMP001',
                    'to_id': 'COMP002',
                    'relationship_type': 'PARTNERS_WITH',
                }
            ],
        )

        with caplog.at_level(logging.INFO, logger='fortune500_kg.data_ingestion_pipeline'):
            result = pipeline.ingest_crawl4ai_data(crawl_data)

        assert result.node_count == 2
        assert result.edge_count == 1

        # The log message must mention both counts
        log_text = ' '.join(caplog.messages)
        assert '2' in log_text
        assert '1' in log_text
        assert any('node' in m.lower() or 'edge' in m.lower() for m in caplog.messages)


class TestValidateDataQuality:
    """Tests for validate_data_quality() (Requirements 15.1-15.5)."""

    def _make_driver_with_records(self, records):
        """Helper: build a mock driver whose session.run() returns given records."""
        driver = Mock()
        session = MagicMock()

        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter(records))
        # list() is called on the result
        mock_result.__iter__ = Mock(return_value=iter(records))
        session.run.return_value = mock_result

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = Mock(return_value=session)
        mock_ctx.__exit__ = Mock(return_value=None)
        driver.session.return_value = mock_ctx
        return driver

    def _make_record(self, id, github_org=None, employee_count=None, revenue_rank=None):
        """Helper: create a dict-like record."""
        rec = Mock()
        rec.__getitem__ = Mock(side_effect=lambda k: {
            'id': id,
            'github_org': github_org,
            'employee_count': employee_count,
            'revenue_rank': revenue_rank,
        }[k])
        return rec

    def test_all_complete_data(self):
        """All companies have complete data → 100% completeness."""
        records = [
            self._make_record('C1', 'org1', 1000, 1),
            self._make_record('C2', 'org2', 2000, 2),
        ]
        driver = self._make_driver_with_records(records)
        pipeline = DataIngestionPipeline(driver)

        report = pipeline.validate_data_quality()

        assert report.total_companies == 2
        assert report.companies_with_complete_data == 2
        assert report.completeness_percentage == 100.0
        assert report.missing_github_org == []
        assert report.missing_employee_count == []
        assert report.missing_revenue_rank == []
        assert report.validation_errors == []

    def test_missing_github_org_identified(self):
        """Companies without github_org appear in missing_github_org list."""
        records = [
            self._make_record('C1', None, 1000, 1),
            self._make_record('C2', 'org2', 2000, 2),
        ]
        driver = self._make_driver_with_records(records)
        pipeline = DataIngestionPipeline(driver)

        report = pipeline.validate_data_quality()

        assert 'C1' in report.missing_github_org
        assert 'C2' not in report.missing_github_org

    def test_missing_employee_count_identified(self):
        """Companies without employee_count appear in missing_employee_count list."""
        records = [
            self._make_record('C1', 'org1', None, 1),
            self._make_record('C2', 'org2', 5000, 2),
        ]
        driver = self._make_driver_with_records(records)
        pipeline = DataIngestionPipeline(driver)

        report = pipeline.validate_data_quality()

        assert 'C1' in report.missing_employee_count
        assert 'C2' not in report.missing_employee_count

    def test_missing_revenue_rank_identified(self):
        """Companies without revenue_rank appear in missing_revenue_rank list."""
        records = [
            self._make_record('C1', 'org1', 1000, None),
            self._make_record('C2', 'org2', 2000, 5),
        ]
        driver = self._make_driver_with_records(records)
        pipeline = DataIngestionPipeline(driver)

        report = pipeline.validate_data_quality()

        assert 'C1' in report.missing_revenue_rank
        assert 'C2' not in report.missing_revenue_rank

    def test_validation_errors_logged_for_missing_fields(self, caplog):
        """Validation failures are logged with company id and missing field name."""
        import logging
        records = [
            self._make_record('C1', None, None, None),
        ]
        driver = self._make_driver_with_records(records)
        pipeline = DataIngestionPipeline(driver)

        with caplog.at_level(logging.WARNING, logger='fortune500_kg.data_ingestion_pipeline'):
            report = pipeline.validate_data_quality()

        # Three ValidationError objects (one per missing field)
        assert len(report.validation_errors) == 3
        error_fields = {e.field_name for e in report.validation_errors}
        assert error_fields == {'github_org', 'employee_count', 'revenue_rank'}

        # Each error must carry the company id
        for err in report.validation_errors:
            assert err.company_id == 'C1'

        # Log messages must mention the company id
        assert any('C1' in m for m in caplog.messages)

    def test_completeness_percentage_calculation(self):
        """Completeness percentage = complete / total * 100."""
        records = [
            self._make_record('C1', 'org1', 1000, 1),   # complete
            self._make_record('C2', None, 2000, 2),      # missing github_org
            self._make_record('C3', 'org3', None, 3),    # missing employee_count
            self._make_record('C4', 'org4', 4000, None), # missing revenue_rank
        ]
        driver = self._make_driver_with_records(records)
        pipeline = DataIngestionPipeline(driver)

        report = pipeline.validate_data_quality()

        assert report.total_companies == 4
        assert report.companies_with_complete_data == 1
        assert abs(report.completeness_percentage - 25.0) < 0.01

    def test_empty_graph_returns_zero_completeness(self):
        """Empty Knowledge Graph → 0% completeness, no errors."""
        driver = self._make_driver_with_records([])
        pipeline = DataIngestionPipeline(driver)

        report = pipeline.validate_data_quality()

        assert report.total_companies == 0
        assert report.completeness_percentage == 0.0
        assert report.validation_errors == []

    def test_report_contains_data_source_statistics(self):
        """Report includes crawl4ai_records, github_api_records, github_api_failures."""
        records = [
            self._make_record('C1', 'org1', 1000, 1),
            self._make_record('C2', None, 2000, 2),
        ]
        driver = self._make_driver_with_records(records)
        pipeline = DataIngestionPipeline(driver)

        report = pipeline.validate_data_quality()

        assert report.crawl4ai_records == 2
        assert report.github_api_records == 1   # only C1 has github_org
        assert report.github_api_failures == 1  # C2 is missing github_org

    def test_validation_error_structure(self):
        """Each ValidationError has company_id, field_name, error_type, error_message."""
        from fortune500_kg.data_models import ValidationError
        records = [
            self._make_record('C1', None, 1000, 1),
        ]
        driver = self._make_driver_with_records(records)
        pipeline = DataIngestionPipeline(driver)

        report = pipeline.validate_data_quality()

        assert len(report.validation_errors) == 1
        err = report.validation_errors[0]
        assert isinstance(err, ValidationError)
        assert err.company_id == 'C1'
        assert err.field_name == 'github_org'
        assert err.error_type == 'missing'
        assert err.error_message != ''
