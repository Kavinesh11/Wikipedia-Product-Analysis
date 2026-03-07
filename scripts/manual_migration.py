"""Manually run migration to debug"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text

# Use absolute path
test_db_path = project_root / "data" / "test_wikipedia_intelligence.db"
db_url = f"sqlite:///{test_db_path.absolute()}"

print(f"Database URL: {db_url}")
print(f"Database path: {test_db_path.absolute()}")
print(f"Database exists: {test_db_path.exists()}")

engine = create_engine(db_url, echo=True)

with engine.connect() as conn:
    # Create a simple test table
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS test_table (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )
    """))
    conn.commit()
    
    # Check if it was created
    result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
    tables = [row[0] for row in result.fetchall()]
    print(f"\nTables after creation: {tables}")

engine.dispose()
