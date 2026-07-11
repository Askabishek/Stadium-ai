"""
Unit tests for StadiumAI tools.
Tests cover database operations, input validation, and core logic.
"""

import os
import sys
import sqlite3
import pytest
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.seed_data import create_database, initialize_database, DB_PATH


class TestDatabaseSetup:
    """Tests for database creation and seeding."""

    def setup_method(self):
        """Setup test database."""
        # Use test database
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        create_database()
        initialize_database()

    def test_database_exists(self):
        """Test that database file is created."""
        assert os.path.exists(DB_PATH)

    def test_stadiums_table_populated(self):
        """Test that stadiums table has data."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stadiums")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 8  # 8 FIFA WC 2026 venues

    def test_facilities_table_populated(self):
        """Test that facilities table has data."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM facilities")
        count = cursor.fetchone()[0]
        conn.close()
        assert count > 0

    def test_crowd_data_table_populated(self):
        """Test that crowd_data table has data."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM crowd_data")
        count = cursor.fetchone()[0]
        conn.close()
        assert count > 0

    def test_matches_table_populated(self):
        """Test that matches table has data."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM matches")
        count = cursor.fetchone()[0]
        conn.close()
        assert count > 0

    def test_transport_table_populated(self):
        """Test that transport table has data."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM transport")
        count = cursor.fetchone()[0]
        conn.close()
        assert count > 0

    def test_stadium_data_integrity(self):
        """Test stadium data has valid coordinates."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT latitude, longitude FROM stadiums")
        for lat, lon in cursor.fetchall():
            assert -90 <= lat <= 90, f"Invalid latitude: {lat}"
            assert -180 <= lon <= 180, f"Invalid longitude: {lon}"
        conn.close()

    def test_capacity_positive(self):
        """Test all stadium capacities are positive."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT capacity FROM stadiums")
        for (capacity,) in cursor.fetchall():
            assert capacity > 0
        conn.close()


class TestNavigationTool:
    """Tests for navigation tool functions."""

    def setup_method(self):
        """Ensure database exists."""
        if not os.path.exists(DB_PATH):
            create_database()
            initialize_database()

    def test_find_facility_returns_list(self):
        """Test find_facility returns a list."""
        from tools.navigation_tool import find_facility
        result = find_facility(1, "Restroom")
        assert isinstance(result, list)

    def test_find_facility_with_accessibility(self):
        """Test accessible facility filter works."""
        from tools.navigation_tool import find_facility
        result = find_facility(1, "Restroom", accessible_only=True)
        for facility in result:
            assert facility["is_accessible"] is True

    def test_find_facility_invalid_stadium(self):
        """Test with non-existent stadium returns empty."""
        from tools.navigation_tool import find_facility
        result = find_facility(999, "Restroom")
        assert result == []

    def test_get_nearest_exit(self):
        """Test nearest exit returns results."""
        from tools.navigation_tool import get_nearest_exit
        result = get_nearest_exit(1, "North Stand")
        assert isinstance(result, list)

    def test_get_stadium_map_data(self):
        """Test map data retrieval."""
        from tools.navigation_tool import get_stadium_map_data
        result = get_stadium_map_data(1)
        assert "stadium" in result
        assert "facilities" in result
        assert result["stadium"]["name"] == "MetLife Stadium"

    def test_get_stadium_map_data_invalid(self):
        """Test map data with invalid stadium."""
        from tools.navigation_tool import get_stadium_map_data
        result = get_stadium_map_data(999)
        assert result == {}


class TestCrowdTool:
    """Tests for crowd management tool."""

    def setup_method(self):
        """Ensure database exists."""
        if not os.path.exists(DB_PATH):
            create_database()
            initialize_database()

    def test_get_current_density_returns_dataframe(self):
        """Test crowd density returns DataFrame."""
        from tools.crowd_tool import get_current_crowd_density
        result = get_current_crowd_density(1)
        assert isinstance(result, pd.DataFrame)

    def test_get_crowd_history(self):
        """Test crowd history retrieval."""
        from tools.crowd_tool import get_crowd_history
        result = get_crowd_history(1, hours=24)
        assert isinstance(result, pd.DataFrame)

    def test_get_overcrowded_zones(self):
        """Test overcrowded zone detection."""
        from tools.crowd_tool import get_overcrowded_zones
        result = get_overcrowded_zones(1)
        assert isinstance(result, list)
        for zone in result:
            assert zone["density_level"] in ["Very High", "Critical"]


class TestMultilingualTool:
    """Tests for multilingual tool."""

    def test_supported_languages(self):
        """Test supported languages list."""
        from tools.multilingual_tool import get_supported_languages
        langs = get_supported_languages()
        assert "en" in langs
        assert "es" in langs
        assert "fr" in langs
        assert len(langs) >= 10

    def test_translate_empty_text(self):
        """Test translation with empty text."""
        from tools.multilingual_tool import translate_text
        result = translate_text("", "es")
        assert result == ""

    def test_translate_same_language(self):
        """Test translation to same language returns original."""
        from tools.multilingual_tool import translate_text
        text = "Hello World"
        result = translate_text(text, "en", "en")
        assert result == text


class TestVoiceTool:
    """Tests for voice tool."""

    def test_speech_to_text_empty_bytes(self):
        """Test STT with empty bytes."""
        from tools.voice_tool import speech_to_text
        result = speech_to_text(b"")
        assert result == ""

    def test_text_to_speech_empty_text(self):
        """Test TTS with empty text raises error."""
        from tools.voice_tool import text_to_speech
        with pytest.raises(ValueError):
            text_to_speech("")

    def test_text_to_speech_whitespace(self):
        """Test TTS with whitespace raises error."""
        from tools.voice_tool import text_to_speech
        with pytest.raises(ValueError):
            text_to_speech("   ")


class TestSQLTool:
    """Tests for SQL tool."""

    def setup_method(self):
        """Ensure database exists."""
        if not os.path.exists(DB_PATH):
            create_database()
            initialize_database()

    def test_execute_query_select(self):
        """Test safe SELECT query execution."""
        from tools.sql_tool import execute_query
        result = execute_query("SELECT COUNT(*) as count FROM stadiums")
        assert not result.empty
        assert result.iloc[0]["count"] == 8

    def test_execute_query_blocks_drop(self):
        """Test that DROP queries are blocked."""
        from tools.sql_tool import execute_query
        result = execute_query("DROP TABLE stadiums")
        assert "error" in result.columns

    def test_execute_query_blocks_delete(self):
        """Test that DELETE queries are blocked."""
        from tools.sql_tool import execute_query
        result = execute_query("DELETE FROM stadiums")
        assert "error" in result.columns

    def test_execute_query_blocks_update(self):
        """Test that UPDATE queries are blocked."""
        from tools.sql_tool import execute_query
        result = execute_query("UPDATE stadiums SET name='hack'")
        assert "error" in result.columns

    def test_get_match_schedule(self):
        """Test match schedule retrieval."""
        from tools.sql_tool import get_match_schedule
        result = get_match_schedule()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_get_feedback_summary(self):
        """Test feedback summary."""
        from tools.sql_tool import get_feedback_summary
        result = get_feedback_summary()
        assert "total_feedback" in result
        assert result["total_feedback"] > 0

    def test_nl_to_sql_empty_question(self):
        """Test NL-to-SQL with empty question."""
        from tools.sql_tool import nl_to_sql
        result = nl_to_sql("")
        assert "Please provide" in result


class TestInputValidation:
    """Tests for input validation and security."""

    def test_sql_injection_prevention(self):
        """Test SQL injection is prevented."""
        from tools.sql_tool import execute_query
        # Attempt SQL injection
        malicious = "SELECT * FROM stadiums; DROP TABLE stadiums;--"
        result = execute_query(malicious)
        # Should either fail or only return SELECT results
        # Table should still exist
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stadiums")
        count = cursor.fetchone()[0]
        conn.close()
        assert count > 0

    def test_facility_search_sanitization(self):
        """Test facility search handles special characters."""
        from tools.navigation_tool import find_facility
        # Should not crash with special characters
        result = find_facility(1, "'; DROP TABLE facilities;--")
        assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
