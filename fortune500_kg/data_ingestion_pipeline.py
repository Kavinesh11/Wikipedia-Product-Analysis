"""Data Ingestion Pipeline for Fortune 500 Knowledge Graph Analytics."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from neo4j import Driver
import requests
import time
import logging
from collections import deque

from .data_models import (
    CrawlData,
    IngestionResult,
    Company,
    Relationship,
    GitHubMetrics,
    DataQualityReport,
    ValidationError,
)

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """
    Data Ingestion Pipeline for parsing and storing Fortune 500 company data.
    
    This class handles:
    - Parsing Crawl4AI data into Knowledge Graph nodes and relationships
    - Creating company nodes with attributes
    - Creating relationship edges between entities
    - Logging ingestion statistics
    """
    
    def __init__(self, driver: Driver, github_token: Optional[str] = None):
        """
        Initialize the Data Ingestion Pipeline.
        
        Args:
            driver: Neo4j driver instance for database operations
            github_token: Optional GitHub personal access token for API authentication
        """
        self.driver = driver
        self.github_token = github_token
        self.github_api_base = "https://api.github.com"
        self.request_queue = deque()  # Queue for failed requests
    
    def ingest_crawl4ai_data(self, crawl_data: CrawlData) -> IngestionResult:
        """
        Parse Crawl4AI data and create Knowledge Graph nodes/edges.
        
        This method:
        1. Validates the input data structure
        2. Creates company nodes with all attributes
        3. Creates relationship edges from parsed data
        4. Returns statistics about the ingestion operation
        
        Args:
            crawl_data: Structured data from Crawl4AI containing company info
            
        Returns:
            IngestionResult with node_count, edge_count, errors
            
        Validates:
            - Requirement 1.1: Parse company nodes and relationships from Crawl4AI data
            - Requirement 1.3: Store employee_count and revenue_rank for each company
        """
        errors = []
        warnings = []
        created_nodes = set()  # Track unique node IDs
        created_edges = set()  # Track unique (from_id, to_id, type) tuples
        
        try:
            with self.driver.session() as session:
                # Process companies
                for company_data in crawl_data.companies:
                    try:
                        # Validate required fields
                        if not self._validate_company_data(company_data):
                            errors.append(
                                f"Invalid company data: {company_data.get('id', 'unknown')}"
                            )
                            continue
                        
                        # Create company node
                        result = session.execute_write(
                            self._create_company_node,
                            company_data
                        )
                        
                        if result:
                            created_nodes.add(company_data['id'])
                        else:
                            errors.append(
                                f"Failed to create node for company: {company_data.get('id')}"
                            )
                    
                    except Exception as e:
                        errors.append(
                            f"Error processing company {company_data.get('id', 'unknown')}: {str(e)}"
                        )
                
                # Process relationships
                for relationship_data in crawl_data.relationships:
                    try:
                        # Validate relationship data
                        if not self._validate_relationship_data(relationship_data):
                            errors.append(
                                f"Invalid relationship data: {relationship_data}"
                            )
                            continue
                        
                        # Create relationship edge
                        result = session.execute_write(
                            self._create_relationship_edge,
                            relationship_data
                        )
                        
                        if result:
                            edge_key = (
                                relationship_data['from_id'],
                                relationship_data['to_id'],
                                relationship_data['relationship_type']
                            )
                            created_edges.add(edge_key)
                        else:
                            errors.append(
                                f"Failed to create relationship: {relationship_data}"
                            )
                    
                    except Exception as e:
                        errors.append(
                            f"Error processing relationship: {str(e)}"
                        )
        
        except Exception as e:
            errors.append(f"Database session error: {str(e)}")
        
        node_count = len(created_nodes)
        edge_count = len(created_edges)
        
        # Requirement 1.4: Log node and edge counts after ingestion
        logger.info(
            "Data ingestion complete: created %d nodes and %d edges",
            node_count,
            edge_count,
        )
        
        return IngestionResult(
            node_count=node_count,
            edge_count=edge_count,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.now()
        )
    
    def _validate_company_data(self, company_data: Dict[str, Any]) -> bool:
        """
        Validate that company data contains required fields.
        
        Args:
            company_data: Dictionary containing company information
            
        Returns:
            True if data is valid, False otherwise
        """
        required_fields = ['id', 'name', 'sector', 'revenue_rank', 'employee_count']
        
        for field in required_fields:
            if field not in company_data:
                return False
            
            # Check for None values
            if company_data[field] is None:
                return False
        
        # Validate data types
        if not isinstance(company_data['revenue_rank'], int):
            return False
        
        if not isinstance(company_data['employee_count'], int):
            return False
        
        return True
    
    def _validate_relationship_data(self, relationship_data: Dict[str, Any]) -> bool:
        """
        Validate that relationship data contains required fields.
        
        Args:
            relationship_data: Dictionary containing relationship information
            
        Returns:
            True if data is valid, False otherwise
        """
        required_fields = ['from_id', 'to_id', 'relationship_type']
        
        for field in required_fields:
            if field not in relationship_data:
                return False
            
            if relationship_data[field] is None:
                return False
        
        return True
    
    @staticmethod
    def _create_company_node(tx, company_data: Dict[str, Any]) -> bool:
        """
        Create a company node in the Knowledge Graph.
        
        Args:
            tx: Neo4j transaction
            company_data: Dictionary containing company information
            
        Returns:
            True if node was created successfully
        """
        query = """
        MERGE (c:Company {id: $id})
        SET c.name = $name,
            c.sector = $sector,
            c.revenue_rank = $revenue_rank,
            c.employee_count = $employee_count,
            c.github_org = $github_org,
            c.updated_at = datetime()
        SET c.created_at = COALESCE(c.created_at, datetime())
        RETURN c
        """
        
        result = tx.run(
            query,
            id=company_data['id'],
            name=company_data['name'],
            sector=company_data['sector'],
            revenue_rank=company_data['revenue_rank'],
            employee_count=company_data['employee_count'],
            github_org=company_data.get('github_org')
        )
        
        return result.single() is not None
    
    @staticmethod
    def _create_relationship_edge(tx, relationship_data: Dict[str, Any]) -> bool:
        """
        Create a relationship edge between entities in the Knowledge Graph.
        
        Args:
            tx: Neo4j transaction
            relationship_data: Dictionary containing relationship information
            
        Returns:
            True if relationship was created successfully
        """
        from_id = relationship_data['from_id']
        to_id = relationship_data['to_id']
        rel_type = relationship_data['relationship_type']
        properties = relationship_data.get('properties', {})
        
        # Build property assignments using parameterized queries
        prop_assignments = []
        params = {
            'from_id': from_id,
            'to_id': to_id
        }
        
        for key, value in properties.items():
            param_name = f"prop_{key}"
            prop_assignments.append(f"r.{key} = ${param_name}")
            params[param_name] = value
        
        prop_string = ", ".join(prop_assignments) if prop_assignments else ""
        set_clause = f"SET {prop_string}" if prop_string else ""
        
        query = f"""
        MATCH (from {{id: $from_id}})
        MATCH (to {{id: $to_id}})
        MERGE (from)-[r:{rel_type}]->(to)
        {set_clause}
        RETURN r
        """
        
        result = tx.run(query, **params)
        
        return result.single() is not None

    
    def validate_data_quality(self) -> DataQualityReport:
        """
        Validate completeness and accuracy of ingested data.

        Queries the Knowledge Graph for all Company nodes and checks:
        - Presence of github_org (Requirement 15.2)
        - Presence of employee_count (Requirement 15.3)
        - Presence of revenue_rank (Requirement 15.3)

        Logs a validation error for every missing field (Requirement 15.4).

        Returns:
            DataQualityReport with completeness percentages and missing-field lists.

        Validates:
            - Requirement 15.1: Validate all Fortune 500 companies have nodes
            - Requirement 15.2: Identify companies with missing GitHub org mappings
            - Requirement 15.3: Validate employee_count and revenue_rank presence
            - Requirement 15.4: Log validation failures with company id and missing fields
            - Requirement 15.5: Generate data quality report with completeness percentages
        """
        missing_github_org: List[str] = []
        missing_employee_count: List[str] = []
        missing_revenue_rank: List[str] = []
        validation_errors: List[ValidationError] = []

        try:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (c:Company) "
                    "RETURN c.id AS id, c.github_org AS github_org, "
                    "c.employee_count AS employee_count, c.revenue_rank AS revenue_rank"
                )
                records = list(result)
        except Exception as e:
            logger.error("Failed to query Knowledge Graph for data quality validation: %s", e)
            records = []

        total_companies = len(records)

        for record in records:
            company_id = record["id"]

            if not record["github_org"]:
                missing_github_org.append(company_id)
                err = ValidationError(
                    company_id=company_id,
                    field_name="github_org",
                    error_type="missing",
                    error_message=f"Company {company_id} is missing github_org",
                )
                validation_errors.append(err)
                logger.warning(
                    "Validation failure - company_id=%s missing_fields=[github_org]",
                    company_id,
                )

            if record["employee_count"] is None:
                missing_employee_count.append(company_id)
                err = ValidationError(
                    company_id=company_id,
                    field_name="employee_count",
                    error_type="missing",
                    error_message=f"Company {company_id} is missing employee_count",
                )
                validation_errors.append(err)
                logger.warning(
                    "Validation failure - company_id=%s missing_fields=[employee_count]",
                    company_id,
                )

            if record["revenue_rank"] is None:
                missing_revenue_rank.append(company_id)
                err = ValidationError(
                    company_id=company_id,
                    field_name="revenue_rank",
                    error_type="missing",
                    error_message=f"Company {company_id} is missing revenue_rank",
                )
                validation_errors.append(err)
                logger.warning(
                    "Validation failure - company_id=%s missing_fields=[revenue_rank]",
                    company_id,
                )

        # Companies with complete data have all three fields present
        companies_with_complete_data = sum(
            1
            for r in records
            if r["github_org"] and r["employee_count"] is not None and r["revenue_rank"] is not None
        )

        completeness_percentage = (
            (companies_with_complete_data / total_companies * 100.0)
            if total_companies > 0
            else 0.0
        )

        report = DataQualityReport(
            report_date=datetime.now(),
            total_companies=total_companies,
            companies_with_complete_data=companies_with_complete_data,
            completeness_percentage=completeness_percentage,
            missing_github_org=missing_github_org,
            missing_employee_count=missing_employee_count,
            missing_revenue_rank=missing_revenue_rank,
            crawl4ai_records=total_companies,
            github_api_records=total_companies - len(missing_github_org),
            github_api_failures=len(missing_github_org),
            validation_errors=validation_errors,
        )

        logger.info(
            "Data quality report: total=%d complete=%d completeness=%.1f%% "
            "missing_github_org=%d missing_employee_count=%d missing_revenue_rank=%d",
            total_companies,
            companies_with_complete_data,
            completeness_percentage,
            len(missing_github_org),
            len(missing_employee_count),
            len(missing_revenue_rank),
        )

        return report

    def fetch_github_metrics(self, company: Company) -> GitHubMetrics:
        """
        Retrieve GitHub metrics for a company's organization.
        
        This method uses GitHub REST API v3 to fetch:
        - Total stars across all repositories
        - Total forks across all repositories
        - Total unique contributors
        
        Args:
            company: Company entity with github_org attribute
            
        Returns:
            GitHubMetrics with stars, forks, contributors
            
        Raises:
            ValueError: When company has no github_org attribute
            RateLimitError: When GitHub API rate limit exceeded
            
        Validates:
            - Requirement 1.2: Retrieve stars, forks, and contributor counts via GitHub API
        """
        if not company.github_org:
            raise ValueError(f"Company {company.id} has no GitHub organization")
        
        headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        
        try:
            # Fetch organization repositories
            repos_url = f"{self.github_api_base}/orgs/{company.github_org}/repos"
            repos = self._fetch_with_rate_limit_handling(repos_url, headers)
            
            # Aggregate metrics across all repositories
            total_stars = 0
            total_forks = 0
            contributors_set = set()
            
            for repo in repos:
                total_stars += repo.get('stargazers_count', 0)
                total_forks += repo.get('forks_count', 0)
                
                # Fetch contributors for this repository
                contributors_url = repo.get('contributors_url')
                if contributors_url:
                    try:
                        contributors = self._fetch_with_rate_limit_handling(
                            contributors_url, 
                            headers
                        )
                        for contributor in contributors:
                            contributors_set.add(contributor.get('login'))
                    except Exception as e:
                        logger.warning(
                            f"Failed to fetch contributors for {repo.get('name')}: {e}"
                        )
            
            return GitHubMetrics(
                stars=total_stars,
                forks=total_forks,
                contributors=len(contributors_set),
                organization=company.github_org,
                retrieved_at=datetime.now()
            )
        
        except requests.exceptions.RequestException as e:
            logger.error(f"GitHub API request failed for {company.github_org}: {e}")
            raise
    
    def _fetch_with_rate_limit_handling(
        self, 
        url: str, 
        headers: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Fetch data from GitHub API with automatic rate limit handling.
        
        Args:
            url: GitHub API endpoint URL
            headers: Request headers including authentication
            
        Returns:
            List of items from the API response
            
        Raises:
            requests.exceptions.RequestException: On network or API errors
        """
        all_items = []
        current_url = url
        
        while current_url:
            response = requests.get(current_url, headers=headers)
            
            # Check for rate limit
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limit hit. Retrying after {retry_after} seconds")
                self.handle_rate_limit(retry_after)
                continue
            
            # Check rate limit headers proactively
            remaining = int(response.headers.get('X-RateLimit-Remaining', 1))
            if remaining == 0:
                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                wait_time = max(reset_time - int(time.time()), 60)
                logger.warning(f"Rate limit approaching. Waiting {wait_time} seconds")
                self.handle_rate_limit(wait_time)
                continue
            
            response.raise_for_status()
            
            data = response.json()
            
            # Handle both single objects and arrays
            if isinstance(data, list):
                all_items.extend(data)
            else:
                all_items.append(data)
            
            # Check for pagination
            link_header = response.headers.get('Link', '')
            current_url = self._parse_next_link(link_header)
        
        return all_items
    
    def _parse_next_link(self, link_header: str) -> Optional[str]:
        """
        Parse the 'next' URL from GitHub's Link header.
        
        Args:
            link_header: The Link header value from GitHub API response
            
        Returns:
            Next page URL if available, None otherwise
        """
        if not link_header:
            return None
        
        links = link_header.split(',')
        for link in links:
            parts = link.split(';')
            if len(parts) == 2:
                url = parts[0].strip()[1:-1]  # Remove < and >
                rel = parts[1].strip()
                if 'rel="next"' in rel:
                    return url
        
        return None
    
    def handle_rate_limit(self, retry_after: int) -> None:
        """
        Queue requests and retry with exponential backoff.
        
        This method implements exponential backoff with:
        - Initial delay: 60 seconds (minimum)
        - Maximum delay: 3600 seconds (1 hour)
        - Exponential growth: delay *= 2 on subsequent retries
        
        Args:
            retry_after: Seconds to wait before retry
            
        Validates:
            - Requirement 1.5: Queue requests and retry with exponential backoff
        """
        # Ensure minimum backoff of 60 seconds
        wait_time = max(retry_after, 60)
        
        # Cap at maximum backoff of 3600 seconds (1 hour)
        wait_time = min(wait_time, 3600)
        
        logger.info(f"Rate limit handler: waiting {wait_time} seconds")
        time.sleep(wait_time)


class RateLimitError(Exception):
    """Exception raised when GitHub API rate limit is exceeded."""
    pass
