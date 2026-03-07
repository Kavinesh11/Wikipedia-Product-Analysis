"""Reputation Monitor

Analyzes edit patterns to assess brand reputation risk and generate alerts.
"""
from typing import List, Optional
from datetime import datetime
import logging

from src.storage.dto import (
    RevisionRecord,
    EditMetrics,
    ReputationScore,
    Alert,
    VandalismMetrics
)

logger = logging.getLogger(__name__)


class ReputationMonitor:
    """Monitor Wikipedia article edit patterns for reputation risks
    
    Analyzes edit velocity, vandalism rates, and editor types to calculate
    reputation risk scores and generate alerts for high-risk situations.
    """
    
    def __init__(self, alert_threshold: float = 0.7):
        """Initialize reputation monitor
        
        Args:
            alert_threshold: Risk score threshold for high-priority alerts (0-1)
        """
        if not 0 <= alert_threshold <= 1:
            raise ValueError("alert_threshold must be between 0 and 1")
        
        self.alert_threshold = alert_threshold
        logger.info(f"ReputationMonitor initialized with alert_threshold={alert_threshold}")
    
    def calculate_reputation_risk(self, edit_metrics: EditMetrics) -> ReputationScore:
        """Calculate composite reputation risk score
        
        Combines edit velocity, vandalism rate, and anonymous edit percentage
        into a weighted risk score normalized to 0-1 range.
        
        Formula:
            risk_score = (0.3 * normalized_velocity + 
                         0.4 * vandalism_rate/100 + 
                         0.3 * anonymous_edit_pct/100)
        
        Args:
            edit_metrics: Aggregated edit pattern metrics
            
        Returns:
            ReputationScore with risk assessment
            
        Validates: Requirements 6.3
        """
        # Normalize edit velocity (cap at 100 edits/hour for normalization)
        normalized_velocity = min(edit_metrics.edit_velocity / 100.0, 1.0)
        
        # Normalize vandalism rate (already a percentage)
        normalized_vandalism = edit_metrics.vandalism_rate / 100.0
        
        # Normalize anonymous edit percentage (already a percentage)
        normalized_anonymous = edit_metrics.anonymous_edit_pct / 100.0
        
        # Weighted combination
        risk_score = (
            0.3 * normalized_velocity +
            0.4 * normalized_vandalism +
            0.3 * normalized_anonymous
        )
        
        # Determine alert level
        if risk_score >= 0.7:
            alert_level = "high"
        elif risk_score >= 0.4:
            alert_level = "medium"
        else:
            alert_level = "low"
        
        logger.debug(
            f"Calculated reputation risk for {edit_metrics.article}: "
            f"score={risk_score:.3f}, level={alert_level}"
        )
        
        return ReputationScore(
            article=edit_metrics.article,
            risk_score=risk_score,
            edit_velocity=edit_metrics.edit_velocity,
            vandalism_rate=edit_metrics.vandalism_rate,
            anonymous_edit_pct=edit_metrics.anonymous_edit_pct,
            alert_level=alert_level,
            timestamp=datetime.now()
        )
    
    def detect_edit_spikes(
        self, 
        edit_velocity: float, 
        baseline: float
    ) -> bool:
        """Detect when edit rate exceeds 3x baseline
        
        Args:
            edit_velocity: Current edit rate (edits per hour)
            baseline: Baseline/normal edit rate (edits per hour)
            
        Returns:
            True if edit velocity exceeds 3x baseline
            
        Validates: Requirements 6.1
        """
        if baseline <= 0:
            # If baseline is zero, any edits could be considered a spike
            return edit_velocity > 0
        
        is_spike = edit_velocity >= (3.0 * baseline)
        
        if is_spike:
            logger.warning(
                f"Edit spike detected: velocity={edit_velocity:.2f}, "
                f"baseline={baseline:.2f}, ratio={edit_velocity/baseline:.2f}x"
            )
        
        return is_spike
    
    def calculate_vandalism_rate(
        self, 
        revisions: List[RevisionRecord]
    ) -> float:
        """Calculate percentage of reverted edits
        
        Formula:
            vandalism_rate = (reverted_edits / total_edits) * 100
        
        Args:
            revisions: List of revision records
            
        Returns:
            Vandalism percentage (0-100)
            
        Validates: Requirements 6.2
        """
        if not revisions:
            return 0.0
        
        total_edits = len(revisions)
        reverted_edits = sum(1 for rev in revisions if rev.is_reverted)
        
        vandalism_rate = (reverted_edits / total_edits) * 100.0
        
        logger.debug(
            f"Calculated vandalism rate: {reverted_edits}/{total_edits} = "
            f"{vandalism_rate:.2f}%"
        )
        
        return vandalism_rate
    
    def generate_alert(
        self, 
        article: str, 
        risk_score: float,
        metadata: Optional[dict] = None
    ) -> Alert:
        """Generate reputation risk alert
        
        Creates an alert with priority based on risk score.
        High-priority alerts are generated when risk exceeds threshold.
        
        Args:
            article: Article title
            risk_score: Reputation risk score (0-1)
            metadata: Optional additional context
            
        Returns:
            Alert object
            
        Validates: Requirements 6.4
        """
        # Determine priority based on risk score
        if risk_score >= 0.7:
            priority = "high"
        elif risk_score >= 0.4:
            priority = "medium"
        else:
            priority = "low"
        
        # Generate alert message
        message = (
            f"Reputation risk detected for article '{article}': "
            f"risk_score={risk_score:.2f} ({priority} priority)"
        )
        
        # Create alert
        alert = Alert(
            alert_id=f"reputation_{article}_{datetime.now().timestamp()}",
            alert_type="reputation_risk",
            priority=priority,
            article=article,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        logger.info(f"Generated {priority} priority alert for {article}")
        
        return alert
    
    def calculate_edit_metrics(
        self,
        article: str,
        revisions: List[RevisionRecord],
        time_window_hours: int = 24
    ) -> EditMetrics:
        """Calculate aggregated edit metrics from revision history
        
        Helper method to compute metrics needed for reputation assessment.
        
        Args:
            article: Article title
            revisions: List of revision records
            time_window_hours: Time window for velocity calculation
            
        Returns:
            EditMetrics with aggregated statistics
        """
        if not revisions:
            return EditMetrics(
                article=article,
                edit_velocity=0.0,
                vandalism_rate=0.0,
                anonymous_edit_pct=0.0,
                total_edits=0,
                reverted_edits=0,
                time_window_hours=time_window_hours
            )
        
        total_edits = len(revisions)
        reverted_edits = sum(1 for rev in revisions if rev.is_reverted)
        anonymous_edits = sum(1 for rev in revisions if rev.editor_type == "anonymous")
        
        # Calculate rates
        vandalism_rate = (reverted_edits / total_edits) * 100.0
        anonymous_edit_pct = (anonymous_edits / total_edits) * 100.0
        edit_velocity = total_edits / time_window_hours
        
        return EditMetrics(
            article=article,
            edit_velocity=edit_velocity,
            vandalism_rate=vandalism_rate,
            anonymous_edit_pct=anonymous_edit_pct,
            total_edits=total_edits,
            reverted_edits=reverted_edits,
            time_window_hours=time_window_hours
        )
    
    def detect_vandalism_signals(
        self,
        revisions: List[RevisionRecord]
    ) -> VandalismMetrics:
        """Identify reverted edits and vandalism patterns
        
        Analyzes revision history to detect potential vandalism signals.
        
        Args:
            revisions: List of revision records
            
        Returns:
            VandalismMetrics with detailed vandalism analysis
            
        Validates: Requirements 2.3
        """
        if not revisions:
            return VandalismMetrics(
                article=revisions[0].article if revisions else "unknown",
                total_edits=0,
                reverted_edits=0,
                vandalism_percentage=0.0,
                revert_patterns=[]
            )
        
        article = revisions[0].article
        total_edits = len(revisions)
        reverted_edits = sum(1 for rev in revisions if rev.is_reverted)
        vandalism_percentage = (reverted_edits / total_edits) * 100.0
        
        # Identify revert patterns
        revert_patterns = []
        for rev in revisions:
            if rev.is_reverted:
                revert_patterns.append({
                    "revision_id": rev.revision_id,
                    "timestamp": rev.timestamp,
                    "editor_type": rev.editor_type,
                    "editor_id": rev.editor_id
                })
        
        logger.debug(
            f"Detected {reverted_edits} reverted edits out of {total_edits} "
            f"for {article} ({vandalism_percentage:.2f}%)"
        )
        
        return VandalismMetrics(
            article=article,
            total_edits=total_edits,
            reverted_edits=reverted_edits,
            vandalism_percentage=vandalism_percentage,
            revert_patterns=revert_patterns
        )
