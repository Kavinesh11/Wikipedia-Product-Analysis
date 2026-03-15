"""Schema management for Neo4j graph database."""

import os
from pathlib import Path
from typing import List, Dict, Any
from neo4j import Driver


class SchemaManager:
    """Manages Neo4j graph schema creation and validation."""
    
    def __init__(self, driver: Driver):
        """
        Initialize schema manager.
        
        Args:
            driver: Neo4j driver instance
        """
        self.driver = driver
        self.schema_dir = Path(__file__).parent / 'schema'
    
    def execute_cypher_file(self, filepath: Path) -> None:
        """
        Execute Cypher statements from a file.
        
        Args:
            filepath: Path to Cypher file
        """
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Split by semicolon and filter out comments and empty lines
        statements = []
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('//'):
                statements.append(line)
        
        cypher = ' '.join(statements)
        
        # Execute each CREATE statement separately
        for statement in cypher.split(';'):
            statement = statement.strip()
            if statement and statement.upper().startswith('CREATE'):
                with self.driver.session() as session:
                    session.run(statement)
    
    def create_constraints(self) -> None:
        """Create uniqueness constraints for node identifiers."""
        constraints_file = self.schema_dir / '01_create_constraints.cypher'
        if constraints_file.exists():
            self.execute_cypher_file(constraints_file)
            print("✓ Constraints created successfully")
        else:
            print(f"Warning: Constraints file not found at {constraints_file}")
    
    def create_indexes(self) -> None:
        """Create indexes for query performance optimization."""
        indexes_file = self.schema_dir / '02_create_indexes.cypher'
        if indexes_file.exists():
            self.execute_cypher_file(indexes_file)
            print("✓ Indexes created successfully")
        else:
            print(f"Warning: Indexes file not found at {indexes_file}")
    
    def initialize_schema(self) -> None:
        """Initialize complete graph schema (constraints and indexes)."""
        print("Initializing Neo4j graph schema...")
        self.create_constraints()
        self.create_indexes()
        print("✓ Schema initialization complete")
    
    def get_constraints(self) -> List[Dict[str, Any]]:
        """
        Get all constraints in the database.
        
        Returns:
            List of constraint information dictionaries
        """
        with self.driver.session() as session:
            result = session.run("SHOW CONSTRAINTS")
            return [dict(record) for record in result]
    
    def get_indexes(self) -> List[Dict[str, Any]]:
        """
        Get all indexes in the database.
        
        Returns:
            List of index information dictionaries
        """
        with self.driver.session() as session:
            result = session.run("SHOW INDEXES")
            return [dict(record) for record in result]
    
    def validate_schema(self) -> Dict[str, bool]:
        """
        Validate that required schema elements exist.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'constraints_exist': False,
            'indexes_exist': False,
            'company_constraint': False,
            'repository_constraint': False,
            'sector_constraint': False,
            'company_sector_index': False,
            'company_revenue_rank_index': False
        }
        
        # Check constraints
        constraints = self.get_constraints()
        constraint_names = [c.get('name', '') for c in constraints]
        
        validation['constraints_exist'] = len(constraints) > 0
        validation['company_constraint'] = any('company_id' in name.lower() for name in constraint_names)
        validation['repository_constraint'] = any('repository_id' in name.lower() for name in constraint_names)
        validation['sector_constraint'] = any('sector_id' in name.lower() for name in constraint_names)
        
        # Check indexes
        indexes = self.get_indexes()
        index_names = [idx.get('name', '') for idx in indexes]
        
        validation['indexes_exist'] = len(indexes) > 0
        validation['company_sector_index'] = any('sector' in name.lower() for name in index_names)
        validation['company_revenue_rank_index'] = any('revenue_rank' in name.lower() for name in index_names)
        
        return validation
    
    def clear_database(self) -> None:
        """
        Clear all nodes and relationships from the database.
        WARNING: This deletes all data!
        """
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✓ Database cleared")
    
    def get_database_stats(self) -> Dict[str, int]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with node and relationship counts
        """
        with self.driver.session() as session:
            # Count nodes by label
            company_count = session.run("MATCH (c:Company) RETURN count(c) as count").single()['count']
            repository_count = session.run("MATCH (r:Repository) RETURN count(r) as count").single()['count']
            sector_count = session.run("MATCH (s:Sector) RETURN count(s) as count").single()['count']
            
            # Count relationships by type
            owns_count = session.run("MATCH ()-[r:OWNS]->() RETURN count(r) as count").single()['count']
            partners_count = session.run("MATCH ()-[r:PARTNERS_WITH]->() RETURN count(r) as count").single()['count']
            acquired_count = session.run("MATCH ()-[r:ACQUIRED]->() RETURN count(r) as count").single()['count']
            belongs_count = session.run("MATCH ()-[r:BELONGS_TO]->() RETURN count(r) as count").single()['count']
            depends_count = session.run("MATCH ()-[r:DEPENDS_ON]->() RETURN count(r) as count").single()['count']
            
            return {
                'companies': company_count,
                'repositories': repository_count,
                'sectors': sector_count,
                'owns_relationships': owns_count,
                'partners_with_relationships': partners_count,
                'acquired_relationships': acquired_count,
                'belongs_to_relationships': belongs_count,
                'depends_on_relationships': depends_count
            }
