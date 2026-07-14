from database import engine
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text("SELECT 1"))
    print("Connection OK, got:", result.scalar())