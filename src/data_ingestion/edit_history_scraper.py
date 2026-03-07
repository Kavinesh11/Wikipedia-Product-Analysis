"""
Edit History Scraper for Wikipedia revision data.

This module provides functionality to extract and analyze Wikipedia edit history,
including editor classification, edit velocity calculation, and vandalism detection.
"""

import asyncio
import re
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from .api_client import WikimediaAPIClient
from ..storage.dto import RevisionRecord, VandalismMetrics, EditMetrics

logger = logging.getLogger(__name__)


class EditHistoryScraper:
    """
    Scraper for Wikipedia edit history with editor classification and vandalism detection.
    
    Features:
    - Fetch revision history from Wikipedia API
    - Classify editors as anonymous or registered
    - Calculate edit velocity over rolling windows
    - Detect vandalism signals through revert detection
    
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.6
    """
    
    def __init__(
        self,
        api_client: Optional[WikimediaAPIClient] = None,
        base_url: str = "https://en.wikipedia.org/w/api.php"
    ):
        """
        Initialize Edit History Scraper.
        
        Args:
            api_client: WikimediaAPIClient instance (creates default if None)
            base_url: Base URL for Wikipedia API
        """
        self.api_client = api_client or WikimediaAPIClient(base_url=base_url)
        self.base_url = base_url
        
        logger.info(f"EditHistoryScraper initialized with base_url={base_url}")
    
    @staticmethod
    def _is_ip_address(editor_id: str) -> bool:
        """
        Check if editor ID is an IP address (anonymous user).
        
        Args:
            editor_id: Editor identifier
            
        Returns:
            True if IP address, False otherwise
        """
        # IPv4 pattern
        ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        # IPv6 pattern (simplified)
        ipv6_pattern = r'^([0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}$'
        
        return bool(re.match(ipv4_pattern, editor_id) or re.match(ipv6_pattern, editor_id))
    
    @staticmethod
    def _classify_editor(editor_id: str) -> str:
        """
        Classify editor as anonymous or registered.
        
        Requirement 2.2: Classify editors as anonymous or registered users
        
        Args:
            editor_id: Editor identifier (username or IP address)
            
        Returns:
            "anonymous" if IP address, "registered" otherwise
        """
        if EditHistoryScraper._is_ip_address(editor_id):
            return "anonymous"
        return "registered"
    
    async def fetch_revisions(
        self,
        article: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 500
    ) -> List[RevisionRecord]:
        """
        Fetch revision history for an article with editor classification.
        
        Requirements:
        - 2.1: Extract edit counts, timestamps, and editor identifiers
        - 2.2: Classify editors as anonymous or registered users
        
        Args:
            article: Article title
            start_date: Start of date range
            end_date: End of date range
            limit: Maximum number of revisions to fetch
            
        Returns:
            List of RevisionRecord objects
            
        Raises:
            aiohttp.ClientError: On API request failure
        """
        logger.info(
            f"Fetching revisions for '{article}' from {start_date} to {end_date}",
            extra={
                "article": article,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "limit": limit
            }
        )
        
        revisions = []
        continue_token = None
        
        while len(revisions) < limit:
            # Build API parameters
            params = {
                "action": "query",
                "format": "json",
                "prop": "revisions",
                "titles": article,
                "rvprop": "ids|timestamp|user|size|comment",
                "rvlimit": min(50, limit - len(revisions)),  # API max is 50 per request
                "rvstart": end_date.isoformat(),
                "rvend": start_date.isoformat(),
                "rvdir": "older"
            }
            
            if continue_token:
                params["rvcontinue"] = continue_token
            
            try:
                # Make API request
                response = await self.api_client.get("", params=params)
                
                # Extract pages from response
                pages = response.get("query", {}).get("pages", {})
                
                if not pages:
                    logger.warning(f"No pages found for article: {article}")
                    break
                
                # Get first (and should be only) page
                page = next(iter(pages.values()))
                
                # Check if page exists
                if "missing" in page:
                    logger.warning(f"Article not found: {article}")
                    break
                
                # Extract revisions
                page_revisions = page.get("revisions", [])
                
                if not page_revisions:
                    logger.debug(f"No more revisions found for {article}")
                    break
                
                # Process each revision
                for rev in page_revisions:
                    try:
                        # Extract revision data
                        revision_id = rev.get("revid")
                        timestamp_str = rev.get("timestamp")
                        editor_id = rev.get("user", "Unknown")
                        size = rev.get("size", 0)
                        comment = rev.get("comment", "")
                        
                        # Parse timestamp
                        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        
                        # Classify editor
                        editor_type = self._classify_editor(editor_id)
                        
                        # Create revision record (is_reverted will be detected later)
                        revision = RevisionRecord(
                            article=article,
                            revision_id=revision_id,
                            timestamp=timestamp,
                            editor_type=editor_type,
                            editor_id=editor_id,
                            is_reverted=False,  # Will be updated by detect_vandalism_signals
                            bytes_changed=0,  # Will be calculated if needed
                            edit_summary=comment
                        )
                        
                        revisions.append(revision)
                        
                    except (KeyError, ValueError) as e:
                        logger.error(
                            f"Error parsing revision: {e}",
                            extra={"revision": rev, "error": str(e)}
                        )
                        continue
                
                # Check for continuation token
                if "continue" in response and len(revisions) < limit:
                    continue_token = response["continue"].get("rvcontinue")
                    logger.debug(f"Fetching more revisions, continue token: {continue_token}")
                else:
                    break
                    
            except Exception as e:
                logger.error(
                    f"Error fetching revisions for {article}: {e}",
                    extra={"article": article, "error": str(e)}
                )
                raise
        
        logger.info(
            f"Fetched {len(revisions)} revisions for '{article}'",
            extra={"article": article, "revision_count": len(revisions)}
        )
        
        return revisions
    
    def calculate_edit_velocity(
        self,
        revisions: List[RevisionRecord],
        window_hours: int = 24
    ) -> float:
        """
        Calculate edit velocity as edits per hour over a rolling window.
        
        Requirement 2.4: Calculate edit velocity as number of edits per hour
        
        Args:
            revisions: List of revision records
            window_hours: Time window in hours (default 24)
            
        Returns:
            Edit velocity (edits per hour)
        """
        if not revisions:
            return 0.0
        
        # Sort revisions by timestamp
        sorted_revisions = sorted(revisions, key=lambda r: r.timestamp)
        
        # Get time range
        start_time = sorted_revisions[0].timestamp
        end_time = sorted_revisions[-1].timestamp
        
        # Calculate actual duration in hours
        duration = (end_time - start_time).total_seconds() / 3600.0
        
        # If duration is less than window, use actual duration
        if duration < window_hours:
            duration = max(duration, 0.001)  # Avoid division by zero
        else:
            duration = window_hours
        
        # Calculate velocity
        edit_count = len(revisions)
        velocity = edit_count / duration
        
        logger.debug(
            f"Edit velocity calculated: {velocity:.2f} edits/hour "
            f"({edit_count} edits over {duration:.2f} hours)"
        )
        
        return velocity
    
    def detect_vandalism_signals(
        self,
        revisions: List[RevisionRecord]
    ) -> VandalismMetrics:
        """
        Detect vandalism signals through revert detection.
        
        Requirement 2.3: Detect reverted edits and flag as potential vandalism
        
        A revert is detected when:
        - Edit summary contains revert keywords ("revert", "undo", "rollback")
        - Or when content is restored to a previous state
        
        Args:
            revisions: List of revision records
            
        Returns:
            VandalismMetrics with revert detection results
        """
        if not revisions:
            return VandalismMetrics(
                article=revisions[0].article if revisions else "Unknown",
                total_edits=0,
                reverted_edits=0,
                vandalism_percentage=0.0,
                revert_patterns=[]
            )
        
        article = revisions[0].article
        total_edits = len(revisions)
        reverted_count = 0
        revert_patterns = []
        
        # Keywords that indicate a revert
        revert_keywords = [
            "revert",
            "undo",
            "undid",
            "rollback",
            "reverted",
            "rv ",
            "restore",
            "restored"
        ]
        
        # Check each revision for revert indicators
        for revision in revisions:
            summary_lower = revision.edit_summary.lower()
            
            # Check if edit summary contains revert keywords
            is_revert = any(keyword in summary_lower for keyword in revert_keywords)
            
            if is_revert:
                revision.is_reverted = True
                reverted_count += 1
                
                revert_patterns.append({
                    "revision_id": revision.revision_id,
                    "timestamp": revision.timestamp.isoformat(),
                    "editor": revision.editor_id,
                    "editor_type": revision.editor_type,
                    "summary": revision.edit_summary
                })
                
                logger.debug(
                    f"Revert detected: revision {revision.revision_id} by {revision.editor_id}",
                    extra={
                        "revision_id": revision.revision_id,
                        "editor": revision.editor_id,
                        "summary": revision.edit_summary
                    }
                )
        
        # Calculate vandalism percentage
        vandalism_percentage = (reverted_count / total_edits * 100) if total_edits > 0 else 0.0
        
        metrics = VandalismMetrics(
            article=article,
            total_edits=total_edits,
            reverted_edits=reverted_count,
            vandalism_percentage=vandalism_percentage,
            revert_patterns=revert_patterns
        )
        
        logger.info(
            f"Vandalism detection for '{article}': "
            f"{reverted_count}/{total_edits} reverted ({vandalism_percentage:.1f}%)",
            extra={
                "article": article,
                "total_edits": total_edits,
                "reverted_edits": reverted_count,
                "vandalism_percentage": vandalism_percentage
            }
        )
        
        return metrics
    
    def calculate_rolling_window_metrics(
        self,
        revisions: List[RevisionRecord],
        windows: List[int] = [24, 168, 720]  # 24h, 7d, 30d in hours
    ) -> Dict[str, EditMetrics]:
        """
        Calculate edit metrics over multiple rolling time windows.
        
        Requirement 2.6: Track edit patterns over rolling time windows (24h, 7d, 30d)
        
        Args:
            revisions: List of revision records
            windows: List of window sizes in hours (default: [24, 168, 720])
            
        Returns:
            Dictionary mapping window label to EditMetrics
        """
        if not revisions:
            return {}
        
        article = revisions[0].article
        metrics_by_window = {}
        
        # Sort revisions by timestamp (newest first)
        sorted_revisions = sorted(revisions, key=lambda r: r.timestamp, reverse=True)
        
        # Get the most recent timestamp as reference
        reference_time = sorted_revisions[0].timestamp
        
        # Window labels
        window_labels = {
            24: "24h",
            168: "7d",
            720: "30d"
        }
        
        for window_hours in windows:
            # Calculate window start time
            window_start = reference_time - timedelta(hours=window_hours)
            
            # Filter revisions within window
            window_revisions = [
                r for r in sorted_revisions
                if r.timestamp >= window_start
            ]
            
            if not window_revisions:
                continue
            
            # Calculate metrics
            total_edits = len(window_revisions)
            reverted_edits = sum(1 for r in window_revisions if r.is_reverted)
            anonymous_edits = sum(1 for r in window_revisions if r.editor_type == "anonymous")
            
            vandalism_rate = (reverted_edits / total_edits * 100) if total_edits > 0 else 0.0
            anonymous_pct = (anonymous_edits / total_edits * 100) if total_edits > 0 else 0.0
            edit_velocity = self.calculate_edit_velocity(window_revisions, window_hours)
            
            metrics = EditMetrics(
                article=article,
                edit_velocity=edit_velocity,
                vandalism_rate=vandalism_rate,
                anonymous_edit_pct=anonymous_pct,
                total_edits=total_edits,
                reverted_edits=reverted_edits,
                time_window_hours=window_hours
            )
            
            window_label = window_labels.get(window_hours, f"{window_hours}h")
            metrics_by_window[window_label] = metrics
            
            logger.debug(
                f"Metrics for {window_label} window: "
                f"velocity={edit_velocity:.2f}, vandalism={vandalism_rate:.1f}%, "
                f"anonymous={anonymous_pct:.1f}%",
                extra={
                    "article": article,
                    "window": window_label,
                    "metrics": {
                        "edit_velocity": edit_velocity,
                        "vandalism_rate": vandalism_rate,
                        "anonymous_edit_pct": anonymous_pct,
                        "total_edits": total_edits
                    }
                }
            )
        
        return metrics_by_window
    
    async def close(self) -> None:
        """Close API client and cleanup resources."""
        await self.api_client.close()
        logger.info("EditHistoryScraper closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
