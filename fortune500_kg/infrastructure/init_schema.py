"""Initialize Neo4j graph schema."""

import sys
from connection import Neo4jConnection
from schema_manager import SchemaManager


def main():
    """Initialize the Neo4j graph schema."""
    print("=" * 60)
    print("Fortune 500 Knowledge Graph - Schema Initialization")
    print("=" * 60)
    
    # Connect to Neo4j
    print("\n1. Connecting to Neo4j...")
    conn = Neo4jConnection()
    
    if not conn.verify_connectivity():
        print("✗ Failed to connect to Neo4j")
        print("  Please ensure Neo4j is running (docker-compose up -d)")
        sys.exit(1)
    
    print("✓ Connected to Neo4j successfully")
    
    # Initialize schema
    print("\n2. Initializing schema...")
    driver = conn.get_driver()
    schema_manager = SchemaManager(driver)
    
    try:
        schema_manager.initialize_schema()
    except Exception as e:
        print(f"✗ Schema initialization failed: {e}")
        conn.close()
        sys.exit(1)
    
    # Validate schema
    print("\n3. Validating schema...")
    validation = schema_manager.validate_schema()
    
    all_valid = all(validation.values())
    
    for check, result in validation.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check}: {result}")
    
    if not all_valid:
        print("\n⚠ Some schema elements are missing")
    else:
        print("\n✓ All schema elements validated successfully")
    
    # Display database stats
    print("\n4. Database statistics:")
    stats = schema_manager.get_database_stats()
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    # Close connection
    conn.close()
    
    print("\n" + "=" * 60)
    print("Schema initialization complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
