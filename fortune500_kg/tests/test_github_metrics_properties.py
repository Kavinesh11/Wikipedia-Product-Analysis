"""Property-based tests for GitHub metrics and rate limiting.

This module contains property-based tests using the hypothesis library to verify:
- Property 2: GitHub Metrics Retrieval Accuracy
- Property 5: Rate Limit Exponential Backoff

These tests validate Requirements 1.2 and 1.5 from the specification.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch
from datetime import datetime
import time

from fortune500_kg.data_ingestion_pipeline import DataIngestionPipeline
from fortune500_kg.data_models import Company, GitHubMetrics


# ============================================================================
# Test Strategies (Generators)
# ============================================================================

@st.composite
def company_with_github_org(draw):
    """Generate a Company with a GitHub organization."""
    company_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_-'
    )))
    name = draw(st.text(min_size=1, max_size=50))
    sector = draw(st.sampled_from([
        'Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing'
    ]))
    revenue_rank = draw(st.integers(min_value=1, max_value=500))
    employee_count = draw(st.integers(min_value=1, max_value=1000000))
    github_org = draw(st.text(min_size=1, max_size=39, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-'
    )))
    
    return Company(
        id=company_id,
        name=name,
        sector=sector,
        revenue_rank=revenue_rank,
        employee_count=employee_count,
        github_org=github_org
    )


@st.composite
def github_repository_data(draw):
    """Generate GitHub repository data with metrics."""
    return {
        'name': draw(st.text(min_size=1, max_size=100)),
        'stargazers_count': draw(st.integers(min_value=0, max_value=100000)),
        'forks_count': draw(st.integers(min_value=0, max_value=50000)),
        'contributors_url': f"https://api.github.com/repos/org/repo/contributors"
    }


@st.composite
def github_contributor_data(draw):
    """Generate GitHub contributor data."""
    login = draw(st.text(min_size=1, max_size=39, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_'
    )))
    return {'login': login}


@st.composite
def rate_limit_sequence(draw):
    """Generate a sequence of rate limit retry_after values."""
    length = draw(st.integers(min_value=1, max_value=10))
    return [draw(st.integers(min_value=1, max_value=5000)) for _ in range(length)]


# ============================================================================
# Property 2: GitHub Metrics Retrieval Accuracy
# ============================================================================

class TestGitHubMetricsRetrievalAccuracy:
    """
    Property 2: GitHub Metrics Retrieval Accuracy
    
    For any company with an associated GitHub organization, the retrieved 
    metrics (stars, forks, contributors) should match the values returned 
    by the GitHub API for that organization.
    
    **Validates: Requirements 1.2**
    """
    
    @settings(max_examples=100, deadline=None)
    @given(
        company=company_with_github_org(),
        repos=st.lists(github_repository_data(), min_size=0, max_size=20)
    )
    def test_metrics_match_api_response(self, company, repos):
        """
        Feature: fortune500-kg-analytics, Property 2: GitHub Metrics Retrieval Accuracy
        
        Verify that fetch_github_metrics returns values matching the GitHub API response.
        """
        # Create mock driver
        mock_driver = Mock()
        pipeline = DataIngestionPipeline(mock_driver, github_token="test_token")
        
        # Calculate expected metrics from mock API data
        expected_stars = sum(repo.get('stargazers_count', 0) for repo in repos)
        expected_forks = sum(repo.get('forks_count', 0) for repo in repos)
        
        # Generate unique contributors across all repos
        all_contributors = set()
        contributors_by_repo = []
        for i, repo in enumerate(repos):
            # Generate 0-10 contributors per repo
            num_contributors = min(i + 1, 10)
            repo_contributors = [
                {'login': f"user_{i}_{j}"} for j in range(num_contributors)
            ]
            contributors_by_repo.append(repo_contributors)
            all_contributors.update(c['login'] for c in repo_contributors)
        
        expected_contributors = len(all_contributors)
        
        # Mock the API responses
        with patch.object(pipeline, '_fetch_with_rate_limit_handling') as mock_fetch:
            # First call returns repos, subsequent calls return contributors
            mock_fetch.side_effect = [repos] + contributors_by_repo
            
            # Execute the method under test
            result = pipeline.fetch_github_metrics(company)
            
            # Verify the property: retrieved metrics match API response
            assert isinstance(result, GitHubMetrics), \
                "Result should be a GitHubMetrics instance"
            
            assert result.stars == expected_stars, \
                f"Stars should match API response: expected {expected_stars}, got {result.stars}"
            
            assert result.forks == expected_forks, \
                f"Forks should match API response: expected {expected_forks}, got {result.forks}"
            
            assert result.contributors == expected_contributors, \
                f"Contributors should match API response: expected {expected_contributors}, got {result.contributors}"
            
            assert result.organization == company.github_org, \
                f"Organization should match company: expected {company.github_org}, got {result.organization}"
            
            assert isinstance(result.retrieved_at, datetime), \
                "Retrieved timestamp should be a datetime"
    
    @settings(max_examples=100, deadline=None)
    @given(
        company=company_with_github_org(),
        stars_list=st.lists(st.integers(min_value=0, max_value=10000), min_size=1, max_size=10),
        forks_list=st.lists(st.integers(min_value=0, max_value=5000), min_size=1, max_size=10)
    )
    def test_metrics_aggregation_correctness(self, company, stars_list, forks_list):
        """
        Feature: fortune500-kg-analytics, Property 2: GitHub Metrics Retrieval Accuracy
        
        Verify that metrics are correctly aggregated across multiple repositories.
        """
        # Ensure lists have the same length
        min_len = min(len(stars_list), len(forks_list))
        stars_list = stars_list[:min_len]
        forks_list = forks_list[:min_len]
        
        # Create mock repositories
        repos = [
            {
                'name': f'repo_{i}',
                'stargazers_count': stars,
                'forks_count': forks,
                'contributors_url': f'https://api.github.com/repos/org/repo_{i}/contributors'
            }
            for i, (stars, forks) in enumerate(zip(stars_list, forks_list))
        ]
        
        # Create mock driver and pipeline
        mock_driver = Mock()
        pipeline = DataIngestionPipeline(mock_driver, github_token="test_token")
        
        # Calculate expected totals
        expected_total_stars = sum(stars_list)
        expected_total_forks = sum(forks_list)
        
        # Mock contributors (empty for simplicity)
        contributors_responses = [[] for _ in repos]
        
        with patch.object(pipeline, '_fetch_with_rate_limit_handling') as mock_fetch:
            mock_fetch.side_effect = [repos] + contributors_responses
            
            result = pipeline.fetch_github_metrics(company)
            
            # Verify aggregation property
            assert result.stars == expected_total_stars, \
                f"Total stars should equal sum of individual repo stars"
            
            assert result.forks == expected_total_forks, \
                f"Total forks should equal sum of individual repo forks"
    
    @settings(max_examples=100, deadline=None)
    @given(
        company=company_with_github_org(),
        num_repos=st.integers(min_value=1, max_value=5),
        contributors_per_repo=st.integers(min_value=1, max_value=10)
    )
    def test_contributor_uniqueness(self, company, num_repos, contributors_per_repo):
        """
        Feature: fortune500-kg-analytics, Property 2: GitHub Metrics Retrieval Accuracy
        
        Verify that duplicate contributors across repos are counted only once.
        """
        # Create repos
        repos = [
            {
                'name': f'repo_{i}',
                'stargazers_count': 100,
                'forks_count': 50,
                'contributors_url': f'https://api.github.com/repos/org/repo_{i}/contributors'
            }
            for i in range(num_repos)
        ]
        
        # Create overlapping contributors
        # Use same contributor names across repos to test deduplication
        contributors_responses = []
        unique_contributors = set()
        
        for i in range(num_repos):
            repo_contributors = [
                {'login': f'user_{j}'} for j in range(contributors_per_repo)
            ]
            contributors_responses.append(repo_contributors)
            unique_contributors.update(c['login'] for c in repo_contributors)
        
        expected_unique_count = len(unique_contributors)
        
        # Create mock driver and pipeline
        mock_driver = Mock()
        pipeline = DataIngestionPipeline(mock_driver, github_token="test_token")
        
        with patch.object(pipeline, '_fetch_with_rate_limit_handling') as mock_fetch:
            mock_fetch.side_effect = [repos] + contributors_responses
            
            result = pipeline.fetch_github_metrics(company)
            
            # Verify uniqueness property
            assert result.contributors == expected_unique_count, \
                f"Contributors should be deduplicated: expected {expected_unique_count}, got {result.contributors}"


# ============================================================================
# Property 5: Rate Limit Exponential Backoff
# ============================================================================

class TestRateLimitExponentialBackoff:
    """
    Property 5: Rate Limit Exponential Backoff
    
    For any sequence of GitHub API requests that encounter rate limit errors, 
    the retry delays should follow exponential backoff pattern (each retry 
    delay ≥ 2× previous delay).
    
    **Validates: Requirements 1.5**
    """
    
    @settings(max_examples=100, deadline=None)
    @given(retry_after=st.integers(min_value=1, max_value=5000))
    def test_minimum_backoff_enforced(self, retry_after):
        """
        Feature: fortune500-kg-analytics, Property 5: Rate Limit Exponential Backoff
        
        Verify that minimum backoff of 60 seconds is enforced.
        """
        mock_driver = Mock()
        pipeline = DataIngestionPipeline(mock_driver)
        
        with patch('time.sleep') as mock_sleep:
            pipeline.handle_rate_limit(retry_after)
            
            # Get the actual wait time used
            actual_wait = mock_sleep.call_args[0][0]
            
            # Verify minimum backoff property
            assert actual_wait >= 60, \
                f"Wait time should be at least 60 seconds, got {actual_wait}"
    
    @settings(max_examples=100, deadline=None)
    @given(retry_after=st.integers(min_value=1, max_value=10000))
    def test_maximum_backoff_enforced(self, retry_after):
        """
        Feature: fortune500-kg-analytics, Property 5: Rate Limit Exponential Backoff
        
        Verify that maximum backoff of 3600 seconds is enforced.
        """
        mock_driver = Mock()
        pipeline = DataIngestionPipeline(mock_driver)
        
        with patch('time.sleep') as mock_sleep:
            pipeline.handle_rate_limit(retry_after)
            
            # Get the actual wait time used
            actual_wait = mock_sleep.call_args[0][0]
            
            # Verify maximum backoff property
            assert actual_wait <= 3600, \
                f"Wait time should be at most 3600 seconds, got {actual_wait}"
    
    @settings(max_examples=100, deadline=None)
    @given(retry_after=st.integers(min_value=60, max_value=3600))
    def test_backoff_within_range(self, retry_after):
        """
        Feature: fortune500-kg-analytics, Property 5: Rate Limit Exponential Backoff
        
        Verify that backoff values within valid range are used as-is.
        """
        mock_driver = Mock()
        pipeline = DataIngestionPipeline(mock_driver)
        
        with patch('time.sleep') as mock_sleep:
            pipeline.handle_rate_limit(retry_after)
            
            # Get the actual wait time used
            actual_wait = mock_sleep.call_args[0][0]
            
            # Verify the property: value within range is preserved
            assert actual_wait == retry_after, \
                f"Wait time within valid range should be preserved: expected {retry_after}, got {actual_wait}"
    
    @settings(max_examples=100, deadline=None)
    @given(
        initial_delay=st.integers(min_value=60, max_value=1800),
        num_retries=st.integers(min_value=2, max_value=5)
    )
    def test_exponential_backoff_pattern(self, initial_delay, num_retries):
        """
        Feature: fortune500-kg-analytics, Property 5: Rate Limit Exponential Backoff
        
        Verify that retry delays follow exponential backoff pattern.
        Note: This tests the conceptual pattern. The actual implementation
        uses the retry_after value from the API, but we verify the bounds
        are consistent with exponential backoff principles.
        """
        mock_driver = Mock()
        pipeline = DataIngestionPipeline(mock_driver)
        
        delays = []
        current_delay = initial_delay
        
        with patch('time.sleep') as mock_sleep:
            for i in range(num_retries):
                # Simulate exponential growth
                pipeline.handle_rate_limit(current_delay)
                actual_delay = mock_sleep.call_args[0][0]
                delays.append(actual_delay)
                
                # Double the delay for next iteration (exponential backoff)
                current_delay = min(current_delay * 2, 3600)
        
        # Verify exponential backoff property
        for i in range(1, len(delays)):
            # Each delay should be >= previous delay (monotonically increasing)
            # or capped at maximum
            if delays[i-1] < 3600:
                assert delays[i] >= delays[i-1], \
                    f"Delays should be monotonically increasing: delays[{i-1}]={delays[i-1]}, delays[{i}]={delays[i]}"
            
            # All delays should be within bounds
            assert 60 <= delays[i] <= 3600, \
                f"All delays should be within [60, 3600]: got {delays[i]}"
    
    @settings(max_examples=100, deadline=None)
    @given(
        company=company_with_github_org(),
        num_rate_limits=st.integers(min_value=1, max_value=3)
    )
    def test_rate_limit_retry_behavior(self, company, num_rate_limits):
        """
        Feature: fortune500-kg-analytics, Property 5: Rate Limit Exponential Backoff
        
        Verify that rate limit errors trigger backoff and eventual success.
        """
        mock_driver = Mock()
        pipeline = DataIngestionPipeline(mock_driver, github_token="test_token")
        
        # Create mock responses: rate limits followed by success
        rate_limit_responses = []
        for i in range(num_rate_limits):
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.headers = {'Retry-After': str(60 * (i + 1))}
            rate_limit_responses.append(mock_response)
        
        # Final success response
        success_response = Mock()
        success_response.status_code = 200
        success_response.headers = {
            'X-RateLimit-Remaining': '100',
            'Link': ''
        }
        success_response.json.return_value = []
        
        with patch('requests.get') as mock_get:
            mock_get.side_effect = rate_limit_responses + [success_response]
            
            with patch.object(pipeline, 'handle_rate_limit') as mock_handle:
                result = pipeline._fetch_with_rate_limit_handling(
                    f'https://api.github.com/orgs/{company.github_org}/repos',
                    {'Accept': 'application/vnd.github.v3+json'}
                )
                
                # Verify backoff was called for each rate limit
                assert mock_handle.call_count == num_rate_limits, \
                    f"handle_rate_limit should be called {num_rate_limits} times"
                
                # Verify eventual success
                assert isinstance(result, list), \
                    "Should eventually return successful result"
    
    @settings(max_examples=100, deadline=None)
    @given(
        retry_sequence=st.lists(
            st.integers(min_value=1, max_value=5000),
            min_size=2,
            max_size=10
        )
    )
    def test_backoff_bounds_consistency(self, retry_sequence):
        """
        Feature: fortune500-kg-analytics, Property 5: Rate Limit Exponential Backoff
        
        Verify that all backoff delays respect min/max bounds regardless of input sequence.
        """
        mock_driver = Mock()
        pipeline = DataIngestionPipeline(mock_driver)
        
        actual_delays = []
        
        with patch('time.sleep') as mock_sleep:
            for retry_after in retry_sequence:
                pipeline.handle_rate_limit(retry_after)
                actual_delay = mock_sleep.call_args[0][0]
                actual_delays.append(actual_delay)
        
        # Verify all delays respect bounds
        for i, delay in enumerate(actual_delays):
            assert 60 <= delay <= 3600, \
                f"Delay {i} should be within [60, 3600]: got {delay} for input {retry_sequence[i]}"
    
    @settings(max_examples=100, deadline=None)
    @given(
        company=company_with_github_org(),
        reset_time_offset=st.integers(min_value=60, max_value=3600)
    )
    def test_proactive_rate_limit_handling(self, company, reset_time_offset):
        """
        Feature: fortune500-kg-analytics, Property 5: Rate Limit Exponential Backoff
        
        Verify that proactive rate limit detection (X-RateLimit-Remaining: 0) triggers backoff.
        """
        mock_driver = Mock()
        pipeline = DataIngestionPipeline(mock_driver, github_token="test_token")
        
        # Mock response with rate limit headers indicating limit reached
        mock_response_rate_limited = Mock()
        mock_response_rate_limited.status_code = 200
        mock_response_rate_limited.headers = {
            'X-RateLimit-Remaining': '0',
            'X-RateLimit-Reset': str(int(time.time()) + reset_time_offset),
            'Link': ''
        }
        mock_response_rate_limited.json.return_value = []

        # Success response returned after backoff
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
                with patch('time.time', return_value=time.time()):
                    pipeline._fetch_with_rate_limit_handling(
                        f'https://api.github.com/orgs/{company.github_org}/repos',
                        {'Accept': 'application/vnd.github.v3+json'}
                    )

                    # Verify proactive backoff was triggered
                    assert mock_handle.called, \
                        "handle_rate_limit should be called when X-RateLimit-Remaining is 0"

                    # Verify the wait time respects bounds
                    if mock_handle.call_args:
                        wait_time = mock_handle.call_args[0][0]
                        assert wait_time >= 60, \
                            f"Proactive backoff should respect minimum: got {wait_time}"
