"""
Navigation Tool — AI-powered stadium navigation and wayfinding.
Helps fans find seats, facilities, exits, and optimal routes.
"""

import os
import sqlite3
from typing import Optional
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


def find_facility(stadium_id: int, facility_type: str, 
                  accessible_only: bool = False) -> list:
    """
    Find facilities of a given type in a stadium.
    
    Args:
        stadium_id: ID of the stadium
        facility_type: Type of facility to find
        accessible_only: If True, only return accessible facilities
    
    Returns:
        List of matching facilities with location details
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT facility_type, icon, zone, floor_level, 
               is_accessible, description, latitude, longitude
        FROM facilities
        WHERE stadium_id = ? AND facility_type LIKE ?
    """
    params = [stadium_id, f"%{facility_type}%"]

    if accessible_only:
        query += " AND is_accessible = 1"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    facilities = []
    for row in rows:
        facilities.append({
            "type": row[0],
            "icon": row[1],
            "zone": row[2],
            "floor_level": row[3],
            "is_accessible": bool(row[4]),
            "description": row[5],
            "latitude": row[6],
            "longitude": row[7]
        })

    return facilities


def get_navigation_directions(stadium_id: int, from_location: str, 
                              to_location: str, 
                              accessibility_needed: bool = False) -> str:
    """
    Generate AI-powered navigation directions within the stadium.
    
    Args:
        stadium_id: ID of the stadium
        from_location: Starting point description
        to_location: Destination description
        accessibility_needed: Whether wheelchair/accessible route is needed
    
    Returns:
        Natural language navigation directions
    """
    client = get_client()

    # Get stadium info
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, city FROM stadiums WHERE id = ?", (stadium_id,))
    stadium = cursor.fetchone()

    # Get relevant facilities
    cursor.execute("""
        SELECT facility_type, zone, floor_level, is_accessible 
        FROM facilities WHERE stadium_id = ?
    """, (stadium_id,))
    facilities = cursor.fetchall()
    conn.close()

    stadium_name = stadium[0] if stadium else "Stadium"
    
    facility_context = "\n".join([
        f"- {f[0]} at {f[1]}, Level {f[2]} {'(Accessible)' if f[3] else ''}"
        for f in facilities[:20]
    ])

    accessibility_note = ""
    if accessibility_needed:
        accessibility_note = """
IMPORTANT: The user requires an accessible route. 
- Only suggest paths with ramps or elevators (no stairs)
- Mention wheelchair-friendly entrances
- Note any accessibility features along the route"""

    prompt = f"""You are a navigation assistant at {stadium_name}. 
Provide clear, step-by-step directions.

Available facilities and zones:
{facility_context}

{accessibility_note}

User wants to go FROM: {from_location}
TO: {to_location}

Provide concise, numbered step-by-step directions. 
Include estimated walking time and any landmarks to look for.
If accessibility is needed, ensure the route is fully accessible."""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Navigation error: {str(e)}"


def get_nearest_exit(stadium_id: int, current_zone: str) -> list:
    """
    Find nearest exits from current zone.
    
    Args:
        stadium_id: ID of the stadium
        current_zone: User's current zone/section
    
    Returns:
        List of nearby exit options
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT facility_type, zone, floor_level, description
        FROM facilities
        WHERE stadium_id = ? AND facility_type = 'Exit Gate'
        ORDER BY zone
    """, (stadium_id,))
    exits = cursor.fetchall()
    conn.close()

    return [
        {
            "type": row[0],
            "zone": row[1],
            "floor_level": row[2],
            "description": row[3]
        }
        for row in exits
    ]


def get_stadium_map_data(stadium_id: int) -> dict:
    """
    Get all map data for a stadium including facilities and zones.
    
    Args:
        stadium_id: ID of the stadium
    
    Returns:
        Dictionary with stadium info and facility locations
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM stadiums WHERE id = ?", (stadium_id,))
    stadium_row = cursor.fetchone()

    cursor.execute("""
        SELECT facility_type, icon, zone, floor_level, 
               is_accessible, latitude, longitude, description
        FROM facilities WHERE stadium_id = ?
    """, (stadium_id,))
    facilities = cursor.fetchall()
    conn.close()

    if not stadium_row:
        return {}

    return {
        "stadium": {
            "id": stadium_row[0],
            "name": stadium_row[1],
            "city": stadium_row[2],
            "country": stadium_row[3],
            "capacity": stadium_row[4],
            "latitude": stadium_row[5],
            "longitude": stadium_row[6]
        },
        "facilities": [
            {
                "type": f[0], "icon": f[1], "zone": f[2],
                "floor_level": f[3], "is_accessible": bool(f[4]),
                "latitude": f[5], "longitude": f[6], "description": f[7]
            }
            for f in facilities
        ]
    }
