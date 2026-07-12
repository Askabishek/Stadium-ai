"""
SQL Tool — Natural language to SQL queries for stadium database.
Enables non-technical staff to query operational data.
"""

import os
import sqlite3
import pandas as pd
from typing import Tuple
from groq import Groq

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "stadium_db.sqlite")


def get_client() -> Groq:
    """Get Groq client with API key from environment."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return Groq(api_key=api_key)


def get_db_connection() -> sqlite3.Connection:
    """Get SQLite database connection."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    return sqlite3.connect(DB_PATH)


def get_schema_info() -> str:
    """Get database schema for LLM context."""
    return """
Database Schema:

TABLE: stadiums
- id (INTEGER PK), name (TEXT), city (TEXT), country (TEXT), 
  capacity (INTEGER), latitude (REAL), longitude (REAL)

TABLE: facilities
- id (INTEGER PK), stadium_id (FK), facility_type (TEXT), icon (TEXT),
  zone (TEXT), floor_level (INTEGER), is_accessible (INTEGER 0/1),
  latitude (REAL), longitude (REAL), description (TEXT)

TABLE: crowd_data
- id (INTEGER PK), stadium_id (FK), zone (TEXT), density_level (TEXT),
  estimated_count (INTEGER), max_capacity (INTEGER), recorded_at (TIMESTAMP)
  density_level values: Low, Moderate, High, Very High, Critical

TABLE: announcements
- id (INTEGER PK), stadium_id (FK), announcement_type (TEXT), message (TEXT),
  language (TEXT), priority (INTEGER 1-5), created_at (TIMESTAMP), is_active (INTEGER)

TABLE: matches
- id (INTEGER PK), stadium_id (FK), team_a (TEXT), team_b (TEXT),
  match_date (TIMESTAMP), match_type (TEXT), status (TEXT)
  match_type values: Group Stage, Round of 16, Quarter Final, Semi Final, Final

TABLE: transport
- id (INTEGER PK), stadium_id (FK), mode (TEXT), details (TEXT),
  is_accessible (INTEGER), estimated_time_min (INTEGER), cost_estimate (TEXT)

TABLE: feedback
- id (INTEGER PK), stadium_id (FK), category (TEXT), message (TEXT),
  sentiment (TEXT: Positive/Neutral/Negative), language (TEXT), created_at (TIMESTAMP)
"""


def nl_to_sql(question: str) -> str:
    """
    Convert natural language question to SQL query using Groq LLM.
    
    Args:
        question: Natural language question about stadium data
    
    Returns:
        Generated SQL query string
    """
    if not question or not question.strip():
        return "SELECT 'Please provide a question' as message"

    client = get_client()
    schema = get_schema_info()

    prompt = f"""You are a SQL expert. Convert the natural language question to a SQLite query.

{schema}

Rules:
- Return ONLY the SQL query, no explanation
- Use proper SQLite syntax
- LIMIT results to 50 rows max
- Use JOINs when needed to get stadium names
- For date filtering, use datetime functions
- Always include meaningful column aliases

Question: {question}

SQL:"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.1
        )
        sql = response.choices[0].message.content.strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql
    except Exception as e:
        return f"-- Error generating SQL: {str(e)}"


def execute_query(sql: str) -> pd.DataFrame:
    """
    Execute SQL query safely and return results as DataFrame.
    
    Args:
        sql: SQL query to execute
    
    Returns:
        Results as pandas DataFrame
    """
    # Basic SQL injection prevention
    dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]
    sql_upper = sql.upper().strip()
    for keyword in dangerous_keywords:
        if sql_upper.startswith(keyword):
            return pd.DataFrame({"error": [f"Unsafe operation blocked: {keyword}"]})

    try:
        conn = get_db_connection()
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


def query_database(question: str) -> Tuple[str, pd.DataFrame]:
    """
    Full pipeline: Natural language question → SQL → Results.
    
    Args:
        question: Natural language question
    
    Returns:
        Tuple of (generated SQL query, results DataFrame)
    """
    sql = nl_to_sql(question)
    if sql.startswith("--"):
        return sql, pd.DataFrame()
    results = execute_query(sql)
    return sql, results


def get_match_schedule(stadium_id: int = None) -> pd.DataFrame:
    """Get match schedule, optionally filtered by stadium."""
    conn = get_db_connection()
    query = """
        SELECT m.team_a, m.team_b, m.match_date, m.match_type, 
               m.status, s.name as stadium
        FROM matches m
        JOIN stadiums s ON m.stadium_id = s.id
    """
    if stadium_id:
        query += f" WHERE m.stadium_id = {int(stadium_id)}"
    query += " ORDER BY m.match_date"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_feedback_summary(stadium_id: int = None) -> dict:
    """Get feedback summary statistics."""
    conn = get_db_connection()
    cursor = conn.cursor()

    where_clause = f"WHERE stadium_id = {int(stadium_id)}" if stadium_id else ""

    cursor.execute(f"SELECT COUNT(*) FROM feedback {where_clause}")
    total = cursor.fetchone()[0]

    cursor.execute(f"""
        SELECT sentiment, COUNT(*) FROM feedback {where_clause} GROUP BY sentiment
    """)
    sentiments = dict(cursor.fetchall())

    cursor.execute(f"""
        SELECT category, COUNT(*) FROM feedback {where_clause} 
        GROUP BY category ORDER BY COUNT(*) DESC LIMIT 5
    """)
    top_categories = cursor.fetchall()

    conn.close()

    return {
        "total_feedback": total,
        "sentiments": sentiments,
        "top_categories": top_categories
    }
