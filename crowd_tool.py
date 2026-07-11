"""
Crowd Management Tool — Real-time crowd density monitoring and predictions.
Helps organizers manage crowd flow and prevent overcrowding.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
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


def get_current_crowd_density(stadium_id: int) -> pd.DataFrame:
    """
    Get current crowd density for all zones in a stadium.
    
    Args:
        stadium_id: ID of the stadium
    
    Returns:
        DataFrame with zone-wise crowd density data
    """
    conn = get_db_connection()
    df = pd.read_sql_query("""
        SELECT zone, density_level, estimated_count, max_capacity, recorded_at
        FROM crowd_data
        WHERE stadium_id = ?
        AND recorded_at = (
            SELECT MAX(recorded_at) FROM crowd_data WHERE stadium_id = ?
        )
        ORDER BY estimated_count DESC
    """, conn, params=(stadium_id, stadium_id))
    conn.close()
    return df


def get_crowd_history(stadium_id: int, hours: int = 24) -> pd.DataFrame:
    """
    Get crowd density history for trend analysis.
    
    Args:
        stadium_id: ID of the stadium
        hours: Number of hours of history to retrieve
    
    Returns:
        DataFrame with historical crowd data
    """
    conn = get_db_connection()
    cutoff = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    df = pd.read_sql_query("""
        SELECT zone, density_level, estimated_count, max_capacity, recorded_at
        FROM crowd_data
        WHERE stadium_id = ? AND recorded_at >= ?
        ORDER BY recorded_at ASC
    """, conn, params=(stadium_id, cutoff))
    conn.close()
    return df


def get_overcrowded_zones(stadium_id: int) -> list:
    """
    Identify zones that are at or near capacity.
    
    Args:
        stadium_id: ID of the stadium
    
    Returns:
        List of zones with critical crowd levels
    """
    df = get_current_crowd_density(stadium_id)
    if df.empty:
        return []

    critical_zones = []
    for _, row in df.iterrows():
        if row["density_level"] in ["Very High", "Critical"]:
            occupancy_pct = (row["estimated_count"] / row["max_capacity"]) * 100
            critical_zones.append({
                "zone": row["zone"],
                "density_level": row["density_level"],
                "estimated_count": row["estimated_count"],
                "max_capacity": row["max_capacity"],
                "occupancy_percentage": round(occupancy_pct, 1)
            })

    return critical_zones


def predict_crowd_flow(stadium_id: int) -> str:
    """
    Use AI to predict crowd flow patterns and suggest interventions.
    
    Args:
        stadium_id: ID of the stadium
    
    Returns:
        AI-generated crowd prediction and recommendations
    """
    client = get_client()

    # Get current and historical data
    current = get_current_crowd_density(stadium_id)
    history = get_crowd_history(stadium_id, hours=6)

    current_summary = current.to_string(index=False) if not current.empty else "No data"
    
    # Calculate trends
    trend_info = ""
    if not history.empty:
        for zone in history["zone"].unique()[:5]:
            zone_data = history[history["zone"] == zone]
            if len(zone_data) > 1:
                first_count = zone_data.iloc[0]["estimated_count"]
                last_count = zone_data.iloc[-1]["estimated_count"]
                change = last_count - first_count
                trend_info += f"- {zone}: {'↑' if change > 0 else '↓'} {abs(change)} people\n"

    prompt = f"""You are a crowd management AI for a FIFA World Cup 2026 stadium.

Current Crowd Density:
{current_summary}

Recent Trends (last 6 hours):
{trend_info if trend_info else "Stable across all zones"}

Based on this data, provide:
1. PREDICTION: What will crowd levels look like in the next 2 hours?
2. HOTSPOTS: Which zones are likely to become overcrowded?
3. RECOMMENDATIONS: Specific actions for staff (open additional gates, redirect flow, deploy more staff)
4. SAFETY ALERT: Any immediate safety concerns?

Be concise and actionable."""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=768,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Prediction error: {str(e)}"


def suggest_alternate_routes(stadium_id: int, destination_zone: str) -> str:
    """
    Suggest alternate routes when a zone is overcrowded.
    
    Args:
        stadium_id: ID of the stadium
        destination_zone: The zone the user wants to reach
    
    Returns:
        AI-generated alternate route suggestions
    """
    client = get_client()
    overcrowded = get_overcrowded_zones(stadium_id)

    overcrowded_info = "\n".join([
        f"- {z['zone']}: {z['occupancy_percentage']}% full ({z['density_level']})"
        for z in overcrowded
    ]) if overcrowded else "No overcrowded zones currently"

    prompt = f"""A fan wants to reach: {destination_zone}

Currently overcrowded zones:
{overcrowded_info}

Suggest the best route to reach {destination_zone} while avoiding overcrowded areas.
Provide 2-3 alternate options with estimated time.
Mention if any route is wheelchair accessible."""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Route suggestion error: {str(e)}"
