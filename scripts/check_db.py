import sqlite3
from pathlib import Path

project_root = Path(__file__).parent.parent
test_db_path = project_root / "data" / "test_wikipedia_intelligence.db"

print(f"Database path: {test_db_path.absolute()}")
print(f"Database exists: {test_db_path.exists()}")

conn = sqlite3.connect(str(test_db_path.absolute()))
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print(f"Tables in database: {tables}")
conn.close()
