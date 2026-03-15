"""Unit tests for GitHub API integration with rate limiting."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import requests

from fortune500_kg.data_ingestion_pipeline import DataIngestionPipeline, RateLimitError
from fortune500_kg.data_models import Company, GitHubMetrics


@pytest.fixture
def mock_driver():
    """Create a mock Neo4j driver."""
    return Mock()


@pytest.fixture
def pipeline(mock_driver):
    """Create a DataIngestionPipeline instance with mock driver."""
    return DataIngestionPipeline(mock_driver, github_token="test_token")


@pytest.fixture
def sample_company():
    """Create a sample company with GitHub organization."""
    return Company(
        id="company_1",
        name="Test Company",
        sector="Technology",
        revenue_rank=1,
        employee_count=10000,
        github_org="test-org"
    )


class TestFetchGitHubMetrics:
    """Test suite for fetch_github_metrics method."""
    
    def test_fetch_metrics_success(self, pipeline, sample_company):
        """Test successful GitHub metrics retrieval."""
        mock_repos = [
            {
                'name': 'repo1',
                'stargazers_count': 100,
                'forks_count': 20,
                'contributors_url': 'https://api.github.com/repos/test-org/repo1/contributors'
            },
            {
                'name': 'repo2',
                'stargazers_count': 50,
                'forks_count': 10,
                'contributors_url': 'https://api.github.com/repos/test-org/repo2/contributors'
            }
        ]
        
        mock_contributors_1 = [
            {'login': 'user1'},
            {'login': 'user2'}
        ]
        
        mock_contributors_2 = [
            {'login': 'user2'},  # Duplicate user
            {'login': 'user3'}
        ]
        
        with patch.object(pipeline, '_fetch_with_rate_limit_handling') as mock_fetch:
            # First call returns repos, subsequent calls return contributors
            mock_fetch.side_effect = [
                mock_repos,
                mock_contributors_1,
                mock_contributors_2
            ]
            
            result = pipeline.fetch_github_metrics(sample_company)
            
            assert isinstance(result, GitHubMetrics)
            assert result.stars == 150  # 100 + 50
            assert result.forks == 30   # 20 + 10
            assert result.contributors == 3  # user1, user2, user3 (unique)
            assert result.organization == "test-org"
    
    def test_fetch_metrics_no_github_org(self, pipeline):
        """Test error when company has no GitHub organization."""
        company = Company(
            id="company_2",
            name="No GitHub Company",
            sector="Finance",
            revenue_rank=2,
            employee_count=5000,
            github_org=None
        )
        
        with pytest.raises(ValueError, match="has no GitHub organization"):
            pipeline.fetch_github_metrics(company)
    
    def test_fetch_metrics_with_token(self, mock_driver, sample_company):
        """Test that GitHub token is included in headers."""
        pipeline = DataIngestionPipeline(mock_driver, github_token="my_token")
        
        with patch.object(pipeline, '_fetch_with_rate_limit_handling') as mock_fetch:
            mock_fetch.return_value = []
            
            pipeline.fetch_github_metrics(sample_company)
            
            # Check that the headers include the token
            call_args = mock_fetch.call_args_list[0]
            headers = call_args[0][1]
            assert headers['Authorization'] == 'token my_token'
    
    def test_fetch_metrics_without_token(self, mock_driver, sample_company):
        """Test GitHub API call without authentication token."""
        pipeline = DataIngestionPipeline(mock_driver, github_token=None)
        
        with patch.object(pipeline, '_fetch_with_rate_limit_handling') as mock_fetch:
            mock_fetch.return_value = []
            
            pipeline.fetch_github_metrics(sample_company)
            
            # Check that Authorization header is not present
            call_args = mock_fetch.call_args_list[0]
            headers = call_args[0][1]
            assert 'Authorization' not in headers
    
    def test_fetch_metrics_handles_missing_contributors_url(self, pipeline, sample_company):
        """Test handling of repositories without contributors_url."""
        mock_repos = [
            {
                'name': 'repo1',
                'stargazers_count': 100,
                'forks_count': 20,
                # No contributors_url
            }
        ]
        
        with patch.object(pipeline, '_fetch_with_rate_limit_handling') as mock_fetch:
            mock_fetch.return_value = mock_repos
            
            result = pipeline.fetch_github_metrics(sample_company)
            
            assert result.stars == 100
            assert result.forks == 20
            assert result.contributors == 0


class TestRateLimitHandling:
    """Test suite for rate limit handling."""
    
    def test_handle_rate_limit_minimum_backoff(self, pipeline):
        """Test that minimum backoff is 60 seconds."""
        with patch('time.sleep') as mock_sleep:
            pipeline.handle_rate_limit(30)  # Request 30 seconds
            mock_sleep.assert_called_once_with(60)  # Should use minimum 60
    
    def test_handle_rate_limit_maximum_backoff(self, pipeline):
        """Test that maximum backoff is 3600 seconds."""
        with patch('time.sleep') as mock_sleep:
            pipeline.handle_rate_limit(5000)  # Request 5000 seconds
            mock_sleep.assert_called_once_with(3600)  # Should cap at 3600
    
    def test_handle_rate_limit_normal_backoff(self, pipeline):
        """Test normal backoff within range."""
        with patch('time.sleep') as mock_sleep:
            pipeline.handle_rate_limit(120)
            mock_sleep.assert_called_once_with(120)
    
    def test_fetch_with_rate_limit_429_response(self, pipeline):
        """Test handling of HTTP 429 rate limit response."""
        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {'Retry-After': '90'}
        
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.headers = {
            'X-RateLimit-Remaining': '100',
            'Link': ''
        }
        mock_response_success.json.return_value = [{'id': 1}]
        
        with patch('requests.get') as mock_get:
            mock_get.side_effect = [mock_response_429, mock_response_success]
            
            with patch.object(pipeline, 'handle_rate_limit') as mock_handle:
                result = pipeline._fetch_with_rate_limit_handling(
                    'https://api.github.com/test',
                    {'Accept': 'application/vnd.github.v3+json'}
                )
                
                # Should have called handle_rate_limit with retry_after value
                mock_handle.assert_called_once_with(90)
                assert len(result) == 1
    
    def test_fetch_with_rate_limit_proactive_check(self, pipeline):
        """Test proactive rate limit checking via headers."""
        mock_response_rate_limited = Mock()
        mock_response_rate_limited.status_code = 200
        mock_response_rate_limited.headers = {
            'X-RateLimit-Remaining': '0',
            'X-RateLimit-Reset': str(int(datetime.now().timestamp()) + 120),
            'Link': ''
        }
        mock_response_rate_limited.json.return_value = []

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.headers = {
            'X-RateLimit-Remaining': '100',
            'Link': ''
        }
        mock_response_success.json.return_value = []

        with patch('requests.get') as mock_get:
            # First call triggers proactive rate limit; second call succeeds
            mock_get.side_effect = [mock_response_rate_limited, mock_response_success]

            with patch.object(pipeline, 'handle_rate_limit') as mock_handle:
                with patch('time.time', return_value=datetime.now().timestamp()):
                    pipeline._fetch_with_rate_limit_handling(
                        'https://api.github.com/test',
                        {'Accept': 'application/vnd.github.v3+json'}
                    )

                    # Should have called handle_rate_limit proactively
                    assert mock_handle.called


class TestPaginationHandling:
    """Test suite for GitHub API pagination."""
    
    def test_parse_next_link_with_pagination(self, pipeline):
        """Test parsing next link from Link header."""
        link_header = (
            '<https://api.github.com/repos?page=2>; rel="next", '
            '<https://api.github.com/repos?page=5>; rel="last"'
        )
        
        next_url = pipeline._parse_next_link(link_header)
        assert next_url == 'https://api.github.com/repos?page=2'
    
    def test_parse_next_link_no_pagination(self, pipeline):
        """Test parsing when no next link exists."""
        link_header = '<https://api.github.com/repos?page=1>; rel="last"'
        
        next_url = pipeline._parse_next_link(link_header)
        assert next_url is None
    
    def test_parse_next_link_empty_header(self, pipeline):
        """Test parsing empty Link header."""
        next_url = pipeline._parse_next_link('')
        assert next_url is None
    
    def test_fetch_with_pagination(self, pipeline):
        """Test fetching multiple pages of results."""
        mock_response_1 = Mock()
        mock_response_1.status_code = 200
        mock_response_1.headers = {
            'X-RateLimit-Remaining': '100',
            'Link': '<https://api.github.com/test?page=2>; rel="next"'
        }
        mock_response_1.json.return_value = [{'id': 1}, {'id': 2}]
        
        mock_response_2 = Mock()
        mock_response_2.status_code = 200
        mock_response_2.headers = {
            'X-RateLimit-Remaining': '99',
            'Link': ''
        }
        mock_response_2.json.return_value = [{'id': 3}]
        
        with patch('requests.get') as mock_get:
            mock_get.side_effect = [mock_response_1, mock_response_2]
            
            result = pipeline._fetch_with_rate_limit_handling(
                'https://api.github.com/test',
                {'Accept': 'application/vnd.github.v3+json'}
            )
            
            assert len(result) == 3
            assert result[0]['id'] == 1
            assert result[1]['id'] == 2
            assert result[2]['id'] == 3


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def test_fetch_metrics_empty_repos(self, pipeline, sample_company):
        """Test handling of organization with no repositories."""
        with patch.object(pipeline, '_fetch_with_rate_limit_handling') as mock_fetch:
            mock_fetch.return_value = []
            
            result = pipeline.fetch_github_metrics(sample_company)
            
            assert result.stars == 0
            assert result.forks == 0
            assert result.contributors == 0
    
    def test_fetch_metrics_missing_counts(self, pipeline, sample_company):
        """Test handling of repositories with missing count fields."""
        mock_repos = [
            {
                'name': 'repo1',
                # Missing stargazers_count and forks_count
                'contributors_url': 'https://api.github.com/repos/test-org/repo1/contributors'
            }
        ]
        
        with patch.object(pipeline, '_fetch_with_rate_limit_handling') as mock_fetch:
            mock_fetch.side_effect = [mock_repos, []]
            
            result = pipeline.fetch_github_metrics(sample_company)
            
            assert result.stars == 0
            assert result.forks == 0
    
    def test_fetch_metrics_network_error(self, pipeline, sample_company):
        """Test handling of network errors."""
        with patch.object(pipeline, '_fetch_with_rate_limit_handling') as mock_fetch:
            mock_fetch.side_effect = requests.exceptions.ConnectionError("Network error")
            
            with pytest.raises(requests.exceptions.ConnectionError):
                pipeline.fetch_github_metrics(sample_company)
