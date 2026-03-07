"""Alert System Module

Handles sending notifications and alerts through multiple channels.
Implements alert deduplication to prevent spam.
"""
import hashlib
import json
import logging
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Set
from dataclasses import asdict

import requests

from src.storage.dto import Alert


logger = logging.getLogger(__name__)


class AlertSystem:
    """System for sending alerts through multiple notification channels
    
    Supports email and webhook notifications with configurable priority levels.
    Implements alert deduplication to prevent sending duplicate alerts within
    a time window.
    """
    
    def __init__(
        self,
        email_config: Optional[Dict] = None,
        webhook_config: Optional[Dict] = None,
        dedup_window_minutes: int = 60
    ):
        """Initialize alert system
        
        Args:
            email_config: Email configuration dict with keys:
                - smtp_host: SMTP server hostname
                - smtp_port: SMTP server port
                - smtp_user: SMTP username
                - smtp_password: SMTP password
                - from_address: Sender email address
                - to_addresses: List of recipient email addresses
            webhook_config: Webhook configuration dict with keys:
                - url: Webhook URL
                - headers: Optional dict of HTTP headers
                - timeout: Request timeout in seconds (default 10)
            dedup_window_minutes: Time window for alert deduplication in minutes
        """
        self.email_config = email_config or {}
        self.webhook_config = webhook_config or {}
        self.dedup_window_minutes = dedup_window_minutes
        
        # Track sent alerts for deduplication
        # Key: alert hash, Value: timestamp when sent
        self._sent_alerts: Dict[str, datetime] = {}
        
        logger.info(
            f"AlertSystem initialized with dedup_window={dedup_window_minutes}min, "
            f"email_enabled={bool(email_config)}, webhook_enabled={bool(webhook_config)}"
        )
    
    def send_alert(
        self,
        alert: Alert,
        channels: Optional[List[str]] = None
    ) -> bool:
        """Send alert through configured notification channels
        
        Args:
            alert: Alert object to send
            channels: List of channels to use ("email", "webhook").
                     If None, uses all configured channels.
        
        Returns:
            True if alert was sent successfully through at least one channel,
            False if alert was deduplicated or all channels failed
        """
        # Check for duplicate alert
        if self._is_duplicate(alert):
            logger.info(
                f"Alert deduplicated: {alert.alert_type} for {alert.article} "
                f"(already sent within {self.dedup_window_minutes} minutes)"
            )
            return False
        
        # Determine which channels to use
        if channels is None:
            channels = []
            if self.email_config:
                channels.append("email")
            if self.webhook_config:
                channels.append("webhook")
        
        if not channels:
            logger.warning("No notification channels configured")
            return False
        
        # Send through each channel
        success = False
        for channel in channels:
            try:
                if channel == "email" and self.email_config:
                    self._send_email(alert)
                    success = True
                    logger.info(f"Alert sent via email: {alert.alert_id}")
                elif channel == "webhook" and self.webhook_config:
                    self._send_webhook(alert)
                    success = True
                    logger.info(f"Alert sent via webhook: {alert.alert_id}")
                else:
                    logger.warning(f"Channel '{channel}' not configured")
            except Exception as e:
                logger.error(
                    f"Failed to send alert via {channel}: {e}",
                    exc_info=True
                )
        
        # Record alert as sent if successful
        if success:
            self._record_sent_alert(alert)
        
        return success
    
    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is a duplicate within deduplication window
        
        Args:
            alert: Alert to check
            
        Returns:
            True if alert is a duplicate, False otherwise
        """
        # Clean up old entries
        self._cleanup_old_alerts()
        
        # Generate hash for alert
        alert_hash = self._generate_alert_hash(alert)
        
        # Check if we've sent this alert recently
        return alert_hash in self._sent_alerts
    
    def _record_sent_alert(self, alert: Alert) -> None:
        """Record that an alert was sent
        
        Args:
            alert: Alert that was sent
        """
        alert_hash = self._generate_alert_hash(alert)
        self._sent_alerts[alert_hash] = datetime.now()
    
    def _cleanup_old_alerts(self) -> None:
        """Remove alerts older than deduplication window"""
        cutoff_time = datetime.now() - timedelta(minutes=self.dedup_window_minutes)
        
        # Remove old entries
        old_hashes = [
            h for h, ts in self._sent_alerts.items()
            if ts < cutoff_time
        ]
        for h in old_hashes:
            del self._sent_alerts[h]
    
    def _generate_alert_hash(self, alert: Alert) -> str:
        """Generate hash for alert deduplication
        
        Uses alert_type, article, and priority to identify duplicates.
        
        Args:
            alert: Alert to hash
            
        Returns:
            Hash string
        """
        # Create hash from key fields
        key_data = f"{alert.alert_type}:{alert.article}:{alert.priority}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _send_email(self, alert: Alert) -> None:
        """Send alert via email
        
        Args:
            alert: Alert to send
            
        Raises:
            Exception: If email sending fails
        """
        # Create message
        msg = MIMEMultipart()
        msg['From'] = self.email_config['from_address']
        msg['To'] = ', '.join(self.email_config['to_addresses'])
        msg['Subject'] = f"[{alert.priority.upper()}] {alert.alert_type}: {alert.article or 'System'}"
        
        # Create email body
        body = self._format_email_body(alert)
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP(
            self.email_config['smtp_host'],
            self.email_config.get('smtp_port', 587)
        ) as server:
            server.starttls()
            if 'smtp_user' in self.email_config and 'smtp_password' in self.email_config:
                server.login(
                    self.email_config['smtp_user'],
                    self.email_config['smtp_password']
                )
            server.send_message(msg)
    
    def _format_email_body(self, alert: Alert) -> str:
        """Format alert as email body
        
        Args:
            alert: Alert to format
            
        Returns:
            Formatted email body
        """
        lines = [
            f"Alert ID: {alert.alert_id}",
            f"Type: {alert.alert_type}",
            f"Priority: {alert.priority}",
            f"Timestamp: {alert.timestamp.isoformat()}",
            "",
            f"Message: {alert.message}",
        ]
        
        if alert.article:
            lines.insert(3, f"Article: {alert.article}")
        
        if alert.metadata:
            lines.extend([
                "",
                "Additional Details:",
                json.dumps(alert.metadata, indent=2)
            ])
        
        return "\n".join(lines)
    
    def _send_webhook(self, alert: Alert) -> None:
        """Send alert via webhook
        
        Args:
            alert: Alert to send
            
        Raises:
            Exception: If webhook request fails
        """
        # Convert alert to dict
        payload = asdict(alert)
        
        # Convert datetime to ISO format
        payload['timestamp'] = alert.timestamp.isoformat()
        
        # Get webhook config
        url = self.webhook_config['url']
        headers = self.webhook_config.get('headers', {})
        timeout = self.webhook_config.get('timeout', 10)
        
        # Set default content type if not specified
        if 'Content-Type' not in headers:
            headers['Content-Type'] = 'application/json'
        
        # Send POST request
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout
        )
        
        # Check response
        response.raise_for_status()
    
    def get_sent_alerts_count(self) -> int:
        """Get count of alerts in deduplication cache
        
        Returns:
            Number of alerts in cache
        """
        self._cleanup_old_alerts()
        return len(self._sent_alerts)
    
    def clear_dedup_cache(self) -> None:
        """Clear alert deduplication cache
        
        Useful for testing or manual override.
        """
        self._sent_alerts.clear()
        logger.info("Alert deduplication cache cleared")
