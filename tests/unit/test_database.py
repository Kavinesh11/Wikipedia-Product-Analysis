"""Unit tests for database utilities"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.storage.database import Database


def test_database_initialization():
    """Test database can be initialized with URL"""
    with patch('src.storage.database.create_engine') as mock_engine:
        with patch('src.storage.database.event'):
            mock_engine.return_value = MagicMock()
            
            db = Database(database_url="postgresql://user:pass@localhost/testdb")
            
            assert db.database_url == "postgresql://user:pass@localhost/testdb"
            assert mock_engine.called


def test_database_session_context_manager():
    """Test database session context manager"""
    with patch('src.storage.database.create_engine'):
        with patch('src.storage.database.event'):
            mock_session = Mock()
            
            db = Database(database_url="postgresql://user:pass@localhost/testdb")
            db.SessionLocal = Mock(return_value=mock_session)
            
            with db.get_session() as session:
                assert session == mock_session
            
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()


def test_database_session_rollback_on_error():
    """Test database session rolls back on error"""
    with patch('src.storage.database.create_engine'):
        with patch('src.storage.database.event'):
            mock_session = Mock()
            mock_session.commit.side_effect = Exception("Test error")
            
            db = Database(database_url="postgresql://user:pass@localhost/testdb")
            db.SessionLocal = Mock(return_value=mock_session)
            
            with pytest.raises(Exception):
                with db.get_session() as session:
                    pass
            
            mock_session.rollback.assert_called_once()
            mock_session.close.assert_called_once()
