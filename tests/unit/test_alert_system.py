"""Unit Tests for Alert System

Tests specific examples, edge cases, and error conditions for alert notifications.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call
import uuid
import smtplib
import requests

from src.utils.alert_system import AlertSystem
from src.storage.dto import Alert


class TestAlertSystem:
    """Test suite for AlertSystem class"""
    
    def test_initialization_default_dedup_window(self):
        """Test alert system initializes with default deduplication window"""
        alert_system = AlertSystem()
        assert alert_system.dedup_window_minutes == 60
        assert alert_system.email_config == {}
        assert alert_system.webhook_config == {}
    
    def test_initialization_custom_dedup_window(self):
        """Test alert system initializes with custom deduplication window"""
        alert_system = AlertSystem(dedup_window_minutes=30)
        assert alert_system.dedup_window_minutes == 30
    
    def test_initialization_with_email_config(self):
        """Test alert system initializes with email configuration"""
        email_config = {
            'smtp_host': 'smtp.example.com',
            'smtp_port': 587,
            'from_address': 'alerts@example.com',
            'to_addresses': ['admin@example.com']
        }
        alert_system = AlertSystem(email_config=email_config)
        assert alert_system.email_config == email_config
    
    def test_initialization_with_webhook_config(self):
        """Test alert system initializes with webhook configuration"""
        webhook_config = {
            'url': 'https://example.com/webhook',
            'timeout': 10
        }
        alert_system = AlertSystem(webhook_config=webhook_config)
        assert alert_system.webhook_config == webhook_config
    
    def test_send_alert_via_email_success(self):
        """Test successful alert delivery via email"""
        with patch('src.utils.alert_system.smtplib.SMTP') as mock_smtp:
            # Configure mock
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            email_config = {
                'smtp_host': 'smtp.example.com',
                'smtp_port': 587,
                'smtp_user': 'alerts@example.com',
                'smtp_password': 'password',
                'from_address': 'alerts@example.com',
                'to_addresses': ['admin@example.com']
            }
            
            alert_system = AlertSystem(email_config=email_config)
            
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                alert_type="pipeline_failure",
                priority="critical",
                article=None,
                message="Test pipeline failed",
                timestamp=datetime.now(),
                metadata={'pipeline': 'test_pipeline'}
            )
            
            result = alert_system.send_alert(alert)
            
            assert result is True
            assert mock_smtp.called
            assert mock_server.starttls.called
            assert mock_server.login.called
            assert mock_server.send_message.called
    
    def test_send_alert_via_webhook_success(self):
        """Test successful alert delivery via webhook"""
        with patch('src.utils.alert_system.requests.post') as mock_post:
            # Configure mock
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            webhook_config = {
                'url': 'https://example.com/webhook',
                'timeout': 10
            }
            
            alert_system = AlertSystem(webhook_config=webhook_config)
            
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                alert_type="reputation_risk",
                priority="high",
                article="Test_Article",
                message="High reputation risk detected",
                timestamp=datetime.now(),
                metadata={'risk_score': 0.85}
            )
            
            result = alert_system.send_alert(alert)
            
            assert result is True
            assert mock_post.called
            
            # Verify webhook payload
            call_args = mock_post.call_args
            assert call_args[1]['json']['alert_type'] == 'reputation_risk'
            assert call_args[1]['json']['priority'] == 'high'
            assert call_args[1]['json']['article'] == 'Test_Article'
    
    def test_send_alert_no_channels_configured(self):
        """Test alert sending fails when no channels are configured"""
        alert_system = AlertSystem()
        
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            alert_type="test_alert",
            priority="low",
            article=None,
            message="Test message",
            timestamp=datetime.now(),
            metadata={}
        )
        
        result = alert_system.send_alert(alert)
        assert result is False
    
    def test_alert_deduplication_blocks_duplicate(self):
        """Test that duplicate alerts are blocked within deduplication window"""
        with patch('src.utils.alert_system.smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            email_config = {
                'smtp_host': 'smtp.example.com',
                'smtp_port': 587,
                'from_address': 'alerts@example.com',
                'to_addresses': ['admin@example.com']
            }
            
            alert_system = AlertSystem(email_config=email_config, dedup_window_minutes=60)
            
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                alert_type="reputation_risk",
                priority="high",
                article="Test_Article",
                message="Test alert",
                timestamp=datetime.now(),
                metadata={}
            )
            
            # Send alert first time
            result1 = alert_system.send_alert(alert)
            assert result1 is True
            
            # Try to send same alert again
            result2 = alert_system.send_alert(alert)
            assert result2 is False
            
            # Verify only one email was sent
            assert mock_server.send_message.call_count == 1
    
    def test_alert_deduplication_allows_different_alerts(self):
        """Test that different alerts are not deduplicated"""
        with patch('src.utils.alert_system.smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            email_config = {
                'smtp_host': 'smtp.example.com',
                'smtp_port': 587,
                'from_address': 'alerts@example.com',
                'to_addresses': ['admin@example.com']
            }
            
            alert_system = AlertSystem(email_config=email_config)
            
            alert1 = Alert(
                alert_id=str(uuid.uuid4()),
                alert_type="reputation_risk",
                priority="high",
                article="Article_1",
                message="Test alert 1",
                timestamp=datetime.now(),
                metadata={}
            )
            
            alert2 = Alert(
                alert_id=str(uuid.uuid4()),
                alert_type="reputation_risk",
                priority="high",
                article="Article_2",  # Different article
                message="Test alert 2",
                timestamp=datetime.now(),
                metadata={}
            )
            
            # Send both alerts
            result1 = alert_system.send_alert(alert1)
            result2 = alert_system.send_alert(alert2)
            
            assert result1 is True
            assert result2 is True
            assert mock_server.send_message.call_count == 2
    
    def test_priority_handling_critical(self):
        """Test that critical priority alerts are handled correctly"""
        with patch('src.utils.alert_system.smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            email_config = {
                'smtp_host': 'smtp.example.com',
                'smtp_port': 587,
                'from_address': 'alerts@example.com',
                'to_addresses': ['admin@example.com']
            }
            
            alert_system = AlertSystem(email_config=email_config)
            
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                alert_type="pipeline_failure",
                priority="critical",
                article=None,
                message="Critical system failure",
                timestamp=datetime.now(),
                metadata={}
            )
            
            result = alert_system.send_alert(alert)
            assert result is True
            
            # Get sent message
            sent_message = mock_server.send_message.call_args[0][0]
            assert "CRITICAL" in sent_message['Subject']
    
    def test_priority_handling_low(self):
        """Test that low priority alerts are handled correctly"""
        with patch('src.utils.alert_system.smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            email_config = {
                'smtp_host': 'smtp.example.com',
                'smtp_port': 587,
                'from_address': 'alerts@example.com',
                'to_addresses': ['admin@example.com']
            }
            
            alert_system = AlertSystem(email_config=email_config)
            
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                alert_type="info",
                priority="low",
                article=None,
                message="Informational message",
                timestamp=datetime.now(),
                metadata={}
            )
            
            result = alert_system.send_alert(alert)
            assert result is True
            
            # Get sent message
            sent_message = mock_server.send_message.call_args[0][0]
            assert "LOW" in sent_message['Subject']
    
    def test_email_failure_handling(self):
        """Test error handling when email sending fails"""
        with patch('src.utils.alert_system.smtplib.SMTP') as mock_smtp:
            # Configure mock to raise exception
            mock_smtp.return_value.__enter__.side_effect = smtplib.SMTPException("Connection failed")
            
            email_config = {
                'smtp_host': 'smtp.example.com',
                'smtp_port': 587,
                'from_address': 'alerts@example.com',
                'to_addresses': ['admin@example.com']
            }
            
            alert_system = AlertSystem(email_config=email_config)
            
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                alert_type="test",
                priority="low",
                article=None,
                message="Test message",
                timestamp=datetime.now(),
                metadata={}
            )
            
            # Should not raise exception, but return False
            result = alert_system.send_alert(alert)
            assert result is False
    
    def test_webhook_failure_handling(self):
        """Test error handling when webhook request fails"""
        with patch('src.utils.alert_system.requests.post') as mock_post:
            # Configure mock to raise exception
            mock_post.side_effect = requests.RequestException("Connection timeout")
            
            webhook_config = {
                'url': 'https://example.com/webhook',
                'timeout': 10
            }
            
            alert_system = AlertSystem(webhook_config=webhook_config)
            
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                alert_type="test",
                priority="low",
                article=None,
                message="Test message",
                timestamp=datetime.now(),
                metadata={}
            )
            
            # Should not raise exception, but return False
            result = alert_system.send_alert(alert)
            assert result is False
    
    def test_channel_selection_email_only(self):
        """Test sending alert through email channel only"""
        with patch('src.utils.alert_system.smtplib.SMTP') as mock_smtp, \
             patch('src.utils.alert_system.requests.post') as mock_webhook:
            
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            email_config = {
                'smtp_host': 'smtp.example.com',
                'smtp_port': 587,
                'from_address': 'alerts@example.com',
                'to_addresses': ['admin@example.com']
            }
            
            webhook_config = {
                'url': 'https://example.com/webhook'
            }
            
            alert_system = AlertSystem(
                email_config=email_config,
                webhook_config=webhook_config
            )
            
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                alert_type="test",
                priority="low",
                article=None,
                message="Test message",
                timestamp=datetime.now(),
                metadata={}
            )
            
            result = alert_system.send_alert(alert, channels=["email"])
            assert result is True
            assert mock_server.send_message.called
            assert not mock_webhook.called
    
    def test_channel_selection_webhook_only(self):
        """Test sending alert through webhook channel only"""
        with patch('src.utils.alert_system.smtplib.SMTP') as mock_smtp, \
             patch('src.utils.alert_system.requests.post') as mock_webhook:
            
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            mock_response = Mock()
            mock_response.status_code = 200
            mock_webhook.return_value = mock_response
            
            email_config = {
                'smtp_host': 'smtp.example.com',
                'smtp_port': 587,
                'from_address': 'alerts@example.com',
                'to_addresses': ['admin@example.com']
            }
            
            webhook_config = {
                'url': 'https://example.com/webhook'
            }
            
            alert_system = AlertSystem(
                email_config=email_config,
                webhook_config=webhook_config
            )
            
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                alert_type="test",
                priority="low",
                article=None,
                message="Test message",
                timestamp=datetime.now(),
                metadata={}
            )
            
            result = alert_system.send_alert(alert, channels=["webhook"])
            assert result is True
            assert not mock_server.send_message.called
            assert mock_webhook.called
    
    def test_get_sent_alerts_count(self):
        """Test getting count of alerts in deduplication cache"""
        with patch('src.utils.alert_system.smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            email_config = {
                'smtp_host': 'smtp.example.com',
                'smtp_port': 587,
                'from_address': 'alerts@example.com',
                'to_addresses': ['admin@example.com']
            }
            
            alert_system = AlertSystem(email_config=email_config)
            
            assert alert_system.get_sent_alerts_count() == 0
            
            # Send an alert
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                alert_type="test",
                priority="low",
                article="Test_Article",
                message="Test message",
                timestamp=datetime.now(),
                metadata={}
            )
            
            alert_system.send_alert(alert)
            assert alert_system.get_sent_alerts_count() == 1
    
    def test_clear_dedup_cache(self):
        """Test clearing deduplication cache"""
        with patch('src.utils.alert_system.smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            email_config = {
                'smtp_host': 'smtp.example.com',
                'smtp_port': 587,
                'from_address': 'alerts@example.com',
                'to_addresses': ['admin@example.com']
            }
            
            alert_system = AlertSystem(email_config=email_config)
            
            # Send an alert
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                alert_type="test",
                priority="low",
                article="Test_Article",
                message="Test message",
                timestamp=datetime.now(),
                metadata={}
            )
            
            alert_system.send_alert(alert)
            assert alert_system.get_sent_alerts_count() == 1
            
            # Clear cache
            alert_system.clear_dedup_cache()
            assert alert_system.get_sent_alerts_count() == 0
            
            # Should be able to send same alert again
            result = alert_system.send_alert(alert)
            assert result is True
    
    def test_email_body_formatting(self):
        """Test email body is formatted correctly"""
        with patch('src.utils.alert_system.smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            email_config = {
                'smtp_host': 'smtp.example.com',
                'smtp_port': 587,
                'from_address': 'alerts@example.com',
                'to_addresses': ['admin@example.com']
            }
            
            alert_system = AlertSystem(email_config=email_config)
            
            alert = Alert(
                alert_id="test-123",
                alert_type="reputation_risk",
                priority="high",
                article="Test_Article",
                message="High risk detected",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                metadata={'risk_score': 0.85, 'velocity': 50}
            )
            
            alert_system.send_alert(alert)
            
            # Get sent message
            sent_message = mock_server.send_message.call_args[0][0]
            body = sent_message.get_payload()[0].get_payload()
            
            # Verify body contains key information
            assert "test-123" in body
            assert "reputation_risk" in body
            assert "high" in body
            assert "Test_Article" in body
            assert "High risk detected" in body
            assert "risk_score" in body
    
    def test_webhook_custom_headers(self):
        """Test webhook requests include custom headers"""
        with patch('src.utils.alert_system.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            webhook_config = {
                'url': 'https://example.com/webhook',
                'headers': {
                    'Authorization': 'Bearer token123',
                    'X-Custom-Header': 'value'
                },
                'timeout': 10
            }
            
            alert_system = AlertSystem(webhook_config=webhook_config)
            
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                alert_type="test",
                priority="low",
                article=None,
                message="Test message",
                timestamp=datetime.now(),
                metadata={}
            )
            
            alert_system.send_alert(alert)
            
            # Verify headers were included
            call_args = mock_post.call_args
            headers = call_args[1]['headers']
            assert headers['Authorization'] == 'Bearer token123'
            assert headers['X-Custom-Header'] == 'value'
            assert headers['Content-Type'] == 'application/json'
