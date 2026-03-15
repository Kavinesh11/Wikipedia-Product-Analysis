"""
Basic example demonstrating DataIngestionPipeline usage.

This example shows how to:
1. Connect to Neo4j
2. Create sample Crawl4AI data
3. Ingest data into the Knowledge Graph
4. View ingestion results
"""

from fortune500_kg import Neo4jConnection, DataIngestionPipeline, CrawlData


def main():
    """Run basic data ingestion example."""
    
    # Step 1: Connect to Neo4j
    print("Connecting to Neo4j...")
    connection = Neo4jConnection()
    
    if not connection.verify_connectivity():
        print("Failed to connect to Neo4j. Please ensure Neo4j is running.")
        return
    
    driver = connection.get_driver()
    print("Connected successfully!")
    
    # Step 2: Create sample Crawl4AI data
    print("\nCreating sample company data...")
    crawl_data = CrawlData(
        companies=[
            {
                'id': 'AAPL',
                'name': 'Apple Inc.',
                'sector': 'Technology',
                'revenue_rank': 3,
                'employee_count': 164000,
                'github_org': 'apple'
            },
            {
                'id': 'MSFT',
                'name': 'Microsoft Corporation',
                'sector': 'Technology',
                'revenue_rank': 14,
                'employee_count': 221000,
                'github_org': 'microsoft'
            },
            {
                'id': 'GOOGL',
                'name': 'Alphabet Inc.',
                'sector': 'Technology',
                'revenue_rank': 11,
                'employee_count': 190234,
                'github_org': 'google'
            }
        ],
        relationships=[
            {
                'from_id': 'AAPL',
                'to_id': 'MSFT',
                'relationship_type': 'PARTNERS_WITH',
                'properties': {
                    'since': '2015-01-01',
                    'partnership_type': 'technology'
                }
            }
        ]
    )
    
    # Step 3: Ingest data into Knowledge Graph
    print("\nIngesting data into Knowledge Graph...")
    pipeline = DataIngestionPipeline(driver)
    result = pipeline.ingest_crawl4ai_data(crawl_data)
    
    # Step 4: Display results
    print("\n" + "="*50)
    print("INGESTION RESULTS")
    print("="*50)
    print(f"Nodes created: {result.node_count}")
    print(f"Edges created: {result.edge_count}")
    print(f"Timestamp: {result.timestamp}")
    
    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors:
            print(f"  - {error}")
    else:
        print("\nNo errors encountered!")
    
    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    # Clean up
    connection.close()
    print("\nConnection closed.")


if __name__ == '__main__':
    main()
