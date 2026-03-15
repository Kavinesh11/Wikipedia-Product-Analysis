"""Neo4j database connection management."""

import os
from typing import Optional
from neo4j import GraphDatabase, Driver
from dotenv import load_dotenv


class Neo4jConnection:
    """Manages Neo4j database connections."""
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (defaults to env variable NEO4J_URI)
            user: Neo4j username (defaults to env variable NEO4J_USER)
            password: Neo4j password (defaults to env variable NEO4J_PASSWORD)
        """
        load_dotenv()
        
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = user or os.getenv('NEO4J_USER', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'fortune500password')
        
        self._driver: Optional[Driver] = None
    
    def connect(self) -> Driver:
        """
        Establish connection to Neo4j database.
        
        Returns:
            Neo4j driver instance
        """
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
        return self._driver
    
    def close(self) -> None:
        """Close the Neo4j connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
    
    def verify_connectivity(self) -> bool:
        """
        Verify connection to Neo4j database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            driver = self.connect()
            driver.verify_connectivity()
            return True
        except Exception as e:
            print(f"Connection verification failed: {e}")
            return False
    
    def get_driver(self) -> Driver:
        """
        Get the Neo4j driver instance.
        
        Returns:
            Neo4j driver instance
        """
        return self.connect()
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
