"""Property-Based Tests for Alert System

Tests correctness properties for alert notifications and deduplication.
"""
import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime, timedelta
from typing import List
import time
from unittest.mock import Mock, patch, MagicMock
import uuid

from src.utils.alert_system import AlertSystem
from src.storage.dto import Alert


# ============================================================================
# Test Strategies
# ============================================================================

@st.composite
def alert_strategy(draw):
    """Generate Alert instances"""
    alert_id = str(uuid.uuid4())
    alert_type = draw(st.sampled_from([
        "reputation_risk",
        "hype_detected",
        "pipeline_failure",
        "edit_spike",
        "vandalism_detected"
    ]))
    priority = draw(st.sampled_from(["low", "medium", "high", "critical"]))
    article = draw(st.one_of(
        st.none(),
        st.text(min_size=1, max_size=100, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_'
        ))
    ))
    message = draw(st.text(min_size=10, max_size=200))
    timestamp = datetime.now()
    metadata = draw(st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.one_of(st.text(max_size=50), st.integers(), st.floats(allow_nan=False)),
        max_size=5
    ))
    
    return Alert(
        alert_id=alert_id,
        alert_type=alert_type,
        priority=priority,
        article=article,
        message=message,
        timestamp=timestamp,
        metadata=metadata
    )


@st.composite
def pipeline_failure_alert_strategy(draw):
    """Generate pipeline failure alerts specifically"""
    alert_id = str(uuid.uuid4())
    pipeline_name = draw(st.sampled_from([
        "pageviews_pipeline",
        "edits_pipeline",
        "crawl_pipeline",
        "analytics_pipeline"
    ]))
    error_message = draw(st.text(min_size=10, max_size=200))
    
    return Alert(
        alert_id=alert_id,
        alert_type="pipeline_failure",
        priority="critical",
        article=None,
        message=f"Pipeline {pipeline_name} failed: {error_message}",
        timestamp=datetime.now(),
        metadata={
            "pipeline_name": pipeline_name,
            "error": error_message
        }
    )


# ============================================================================
# Property 54: Failure Notifications
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 54: Failure Notifications
@given(alert=pipeline_failure_alert_strategy())
@settings(max_examples=5, deadline=None)
def test_property_54_failure_notifications(alert):
    """
    Property 54: For any pipeline failure, the System should send a notification
    to configured administrators within 1 minute.
    
    **Validates: Requirements 11.6**
    
    This test verifies that:
    1. Pipeline failure alerts are sent successfully
    2. Alerts are sent through configured channels
    3. The notification is sent promptly (within reasonable time)
    """
    # Mock email and webhook sending
    with patch('src.utils.alert_system.smtplib.SMTP') as mock_smtp, \
         patch('src.utils.alert_system.requests.post') as mock_webhook:
        
        # Configure mock SMTP
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # Configure mock webhook
        mock_response = Mock()
        mock_response.status_code = 200
        mock_webhook.return_value = mock_response
        
        # Create alert system with both channels
        email_config = {
            'smtp_host': 'smtp.example.com',
            'smtp_port': 587,
            'smtp_user': 'alerts@example.com',
            'smtp_password': 'password',
            'from_address': 'alerts@example.com',
            'to_addresses': ['admin@example.com']
        }
        
        webhook_config = {
            'url': 'https://example.com/webhook',
            'timeout': 10
        }
        
        alert_system = AlertSystem(
            email_config=email_config,
            webhook_config=webhook_config
        )
        
        # Record start time
        start_time = time.time()
        
        # Send alert
        result = alert_system.send_alert(alert)
        
        # Record end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Verify alert was sent successfully
        assert result is True, "Pipeline failure alert should be sent successfully"
        
        # Verify notification was sent within reasonable time (1 second for unit test)
        # In production, this would be within 1 minute as per requirement
        assert elapsed_time < 1.0, \
            f"Alert should be sent promptly, took {elapsed_time:.2f}s"
        
        # Verify email was sent
        assert mock_smtp.called, "Email notification should be attempted"
        assert mock_server.send_message.called, "Email should be sent"
        
        # Verify webhook was sent
        assert mock_webhook.called, "Webhook notification should be attempted"
        
        # Verify webhook payload contains alert data
        webhook_call_args = mock_webhook.call_args
        payload = webhook_call_args[1]['json']
        assert payload['alert_type'] == 'pipeline_failure'
        assert payload['priority'] == 'critical'
        assert 'pipeline_name' in payload['metadata']


# ============================================================================
# Additional Alert System Properties
# ============================================================================

# Test alert deduplication
@given(alert=alert_strategy())
@settings(max_examples=5, deadline=None)
def test_alert_deduplication(alert):
    """
    Property: Sending the same alert multiple times within the deduplication
    window should only result in one notification being sent.
    
    This prevents alert spam.
    """
    with patch('src.utils.alert_system.smtplib.SMTP') as mock_smtp, \
         patch('src.utils.alert_system.requests.post') as mock_webhook:
        
        # Configure mocks
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        mock_response = Mock()
        mock_response.status_code = 200
        mock_webhook.return_value = mock_response
        
        # Create alert system with short dedup window
        email_config = {
            'smtp_host': 'smtp.example.com',
            'smtp_port': 587,
            'from_address': 'alerts@example.com',
            'to_addresses': ['admin@example.com']
        }
        
        alert_system = AlertSystem(
            email_config=email_config,
            dedup_window_minutes=60
        )
        
        # Send alert first time
        result1 = alert_system.send_alert(alert)
        assert result1 is True, "First alert should be sent"
        
        # Send same alert again immediately
        result2 = alert_system.send_alert(alert)
        assert result2 is False, "Duplicate alert should be blocked"
        
        # Verify only one email was sent
        assert mock_server.send_message.call_count == 1, \
            "Only one email should be sent for duplicate alerts"


# Test priority handling
@given(
    priority=st.sampled_from(["low", "medium", "high", "critical"]),
    alert_type=st.sampled_from(["reputation_risk", "pipeline_failure", "hype_detected"])
)
@settings(max_examples=5, deadline=None)
def test_alert_priority_handling(priority, alert_type):
    """
    Property: Alerts should be sent with the correct priority level,
    and the priority should be reflected in the notification.
    """
    with patch('src.utils.alert_system.smtplib.SMTP') as mock_smtp:
        
        # Configure mock
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        email_config = {
            'smtp_host': 'smtp.example.com',
            'smtp_port': 587,
            'from_address': 'alerts@example.com',
            'to_addresses': ['admin@example.com']
        }
        
        alert_system = AlertSystem(email_config=email_config)
        
        # Create alert with specific priority
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            priority=priority,
            article="Test_Article",
            message=f"Test {alert_type} alert",
            timestamp=datetime.now(),
            metadata={}
        )
        
        # Send alert
        result = alert_system.send_alert(alert)
        assert result is True, "Alert should be sent successfully"
        
        # Verify email was sent
        assert mock_server.send_message.called
        
        # Get the message that was sent
        sent_message = mock_server.send_message.call_args[0][0]
        
        # Verify priority is in subject line
        assert priority.upper() in sent_message['Subject'], \
            f"Priority '{priority}' should be in email subject"


# Test multiple notification channels
@given(alert=alert_strategy())
@settings(max_examples=5, deadline=None)
def test_multiple_notification_channels(alert):
    """
    Property: When multiple notification channels are configured,
    alerts should be sent through all channels.
    """
    with patch('src.utils.alert_system.smtplib.SMTP') as mock_smtp, \
         patch('src.utils.alert_system.requests.post') as mock_webhook:
        
        # Configure mocks
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        mock_response = Mock()
        mock_response.status_code = 200
        mock_webhook.return_value = mock_response
        
        # Create alert system with both channels
        email_config = {
            'smtp_host': 'smtp.example.com',
            'smtp_port': 587,
            'from_address': 'alerts@example.com',
            'to_addresses': ['admin@example.com']
        }
        
        webhook_config = {
            'url': 'https://example.com/webhook',
            'timeout': 10
        }
        
        alert_system = AlertSystem(
            email_config=email_config,
            webhook_config=webhook_config
        )
        
        # Send alert
        result = alert_system.send_alert(alert)
        assert result is True, "Alert should be sent successfully"
        
        # Verify both channels were used
        assert mock_server.send_message.called, "Email should be sent"
        assert mock_webhook.called, "Webhook should be called"


# Test channel selection
@given(
    alert=alert_strategy(),
    channel=st.sampled_from(["email", "webhook"])
)
@settings(max_examples=5, deadline=None)
def test_channel_selection(alert, channel):
    """
    Property: When specific channels are requested, only those channels
    should be used for sending alerts.
    """
    with patch('src.utils.alert_system.smtplib.SMTP') as mock_smtp, \
         patch('src.utils.alert_system.requests.post') as mock_webhook:
        
        # Configure mocks
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        mock_response = Mock()
        mock_response.status_code = 200
        mock_webhook.return_value = mock_response
        
        # Create alert system with both channels
        email_config = {
            'smtp_host': 'smtp.example.com',
            'smtp_port': 587,
            'from_address': 'alerts@example.com',
            'to_addresses': ['admin@example.com']
        }
        
        webhook_config = {
            'url': 'https://example.com/webhook',
            'timeout': 10
        }
        
        alert_system = AlertSystem(
            email_config=email_config,
            webhook_config=webhook_config
        )
        
        # Send alert through specific channel
        result = alert_system.send_alert(alert, channels=[channel])
        assert result is True, f"Alert should be sent via {channel}"
        
        # Verify only the requested channel was used
        if channel == "email":
            assert mock_server.send_message.called, "Email should be sent"
            assert not mock_webhook.called, "Webhook should not be called"
        else:
            assert not mock_server.send_message.called, "Email should not be sent"
            assert mock_webhook.called, "Webhook should be called"
