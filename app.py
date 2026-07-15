"""
StadiumAI — FIFA World Cup 2026 GenAI Stadium Operations Platform.

A multimodal AI-powered solution that enhances stadium operations and
the overall tournament experience for fans, organizers, volunteers,
and venue staff during the FIFA World Cup 2026.

Author: Abishek
License: MIT
"""

import os
import sys
import logging
from typing import Optional

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from data.seed_data import create_database, initialize_database
from tools.sql_tool import get_match_schedule, get_feedback_summary, query_database
from tools.crowd_tool import (
    get_current_crowd_density,
    get_overcrowded_zones,
    predict_crowd_flow,
)
from tools.navigation_tool import (
    find_facility,
    get_navigation_directions,
    get_stadium_map_data,
)
from tools.multilingual_tool import (
    translate_text,
    get_supported_languages,
    generate_audio_announcement,
    translate_announcement,
)
from tools.rag_tool import index_stadium_data, semantic_search
from tools.voice_tool import speech_to_text, text_to_speech
from pipeline import (
    process_fan_query,
    process_voice_query,
    generate_operational_insights,
    generate_emergency_response,
)

# --- Constants ---
APP_TITLE = "StadiumAI — FIFA World Cup 2026"
APP_ICON = "⚽"
STADIUMS_LIST = [
    "MetLife Stadium",
    "AT&T Stadium",
    "SoFi Stadium",
    "Estadio Azteca",
    "Hard Rock Stadium",
    "Lumen Field",
    "BMO Field",
    "NRG Stadium",
]
FACILITY_TYPES = [
    "Restroom",
    "Food Court",
    "First Aid",
    "Information Desk",
    "Merchandise Store",
    "ATM",
    "Prayer Room",
    "Baby Care Room",
    "Wheelchair Access Point",
    "Water Station",
    "Charging Station",
    "Exit Gate",
]
EMERGENCY_TYPES = [
    "Medical Emergency",
    "Fire",
    "Structural Issue",
    "Crowd Crush",
    "Severe Weather",
    "Security Threat",
    "Power Failure",
]
DENSITY_COLORS = {
    "Low": "#4CAF50",
    "Moderate": "#FFC107",
    "High": "#FF9800",
    "Very High": "#FF5722",
    "Critical": "#B71C1C",
}
SENTIMENT_COLORS = {
    "Positive": "#4CAF50",
    "Neutral": "#FFC107",
    "Negative": "#F44336",
}


def configure_page() -> None:
    """Configure Streamlit page settings and inject custom CSS."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_get_custom_css(), unsafe_allow_html=True)


def _get_custom_css() -> str:
    """Return custom CSS for styling and accessibility."""
    return """
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1A237E;
            text-align: center;
            margin-bottom: 0.3rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #283593;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .alert-critical {
            background: #FFEBEE;
            border-left: 4px solid #C62828;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 0.8rem;
        }
        .alert-warning {
            background: #FFF8E1;
            border-left: 4px solid #F57F17;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 0.8rem;
        }
        /* Accessibility: High contrast focus indicators */
        *:focus {
            outline: 3px solid #1565C0 !important;
            outline-offset: 2px;
        }
        /* Accessibility: Minimum touch target size (WCAG 2.5.5) */
        button, .stButton > button {
            min-height: 44px;
            min-width: 44px;
        }
        /* Skip navigation link for screen readers */
        .skip-link {
            position: absolute;
            top: -40px;
            left: 0;
            background: #1565C0;
            color: white;
            padding: 8px;
            z-index: 100;
        }
        .skip-link:focus { top: 0; }
    </style>
    <a href="#main-content" class="skip-link" aria-label="Skip to main content">
        Skip to main content
    </a>
    """


@st.cache_resource
def initialize_app() -> int:
    """
    Initialize database and vector index on first run.

    Returns:
        Number of documents indexed in the vector store.
    """
    logger.info("Initializing StadiumAI database and vector index...")
    create_database()
    initialize_database()
    count = index_stadium_data()
    logger.info("Initialization complete. %d documents indexed.", count)
    return count


def validate_api_key() -> bool:
    """
    Check if GROQ_API_KEY is configured.

    Returns:
        True if API key is available, False otherwise.
    """
    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY is not configured. Please set it in environment variables.")
        return False
    return True


def render_sidebar() -> tuple:
    """
    Render the sidebar navigation and settings.

    Returns:
        Tuple of (selected_page, stadium_id, accessibility_mode).
    """
    with st.sidebar:
        st.markdown("## ⚽ StadiumAI")
        st.markdown("**FIFA World Cup 2026**")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            [
                "🏠 Dashboard",
                "🧭 Fan Navigator",
                "👥 Crowd Monitor",
                "🌐 Multilingual Hub",
                "🎤 Voice Assistant",
                "🔍 Smart Search",
                "📊 Operations Center",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 🏟️ Select Stadium")
        selected_stadium = st.selectbox(
            "Stadium",
            STADIUMS_LIST,
            label_visibility="collapsed",
        )
        stadium_id = STADIUMS_LIST.index(selected_stadium) + 1

        st.markdown("---")
        accessibility_mode = st.toggle(
            "♿ Accessibility Mode",
            value=False,
            help="Enable to prioritize accessible routes and facilities",
        )

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; color:#666; font-size:0.8rem;'>"
            "Powered by<br><b>Groq Llama 4 Scout</b><br>"
            "Whisper • Chroma • Streamlit</div>",
            unsafe_allow_html=True,
        )

    return page, stadium_id, accessibility_mode


def render_page_header(icon: str, title: str, subtitle: str) -> None:
    """
    Render a consistent page header with accessibility landmark.

    Args:
        icon: Emoji icon for the page.
        title: Main page title.
        subtitle: Descriptive subtitle.
    """
    st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="main-header">{icon} {title}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="sub-header">{subtitle}</p>',
        unsafe_allow_html=True,
    )


# ==================== PAGE RENDERERS ====================


def render_dashboard(stadium_id: int) -> None:
    """Render the main dashboard with key metrics and alerts."""
    render_page_header("⚽", "StadiumAI", "AI-Powered Stadium Operations — FIFA World Cup 2026")

    crowd_data = get_current_crowd_density(stadium_id)
    overcrowded = get_overcrowded_zones(stadium_id)
    matches = get_match_schedule(stadium_id)
    feedback = get_feedback_summary(stadium_id)

    _render_dashboard_metrics(crowd_data, overcrowded, matches, feedback)
    st.markdown("---")
    _render_dashboard_charts(crowd_data, matches)
    _render_dashboard_alerts(overcrowded)


def _render_dashboard_metrics(
    crowd_data: pd.DataFrame,
    overcrowded: list,
    matches: pd.DataFrame,
    feedback: dict,
) -> None:
    """Render the key metric cards on the dashboard."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_fans = int(crowd_data["estimated_count"].sum()) if not crowd_data.empty else 0
        st.metric("👥 Current Fans", f"{total_fans:,}")
    with col2:
        st.metric("🚨 Crowded Zones", len(overcrowded))
    with col3:
        upcoming = len(matches[matches["status"] == "Scheduled"]) if not matches.empty else 0
        st.metric("📅 Upcoming Matches", upcoming)
    with col4:
        st.metric("💬 Fan Feedback", feedback.get("total_feedback", 0))


def _render_dashboard_charts(crowd_data: pd.DataFrame, matches: pd.DataFrame) -> None:
    """Render crowd overview chart and match schedule table."""
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("👥 Crowd Density Overview")
        if not crowd_data.empty:
            fig = px.bar(
                crowd_data.head(10),
                x="zone",
                y="estimated_count",
                color="density_level",
                color_discrete_map=DENSITY_COLORS,
                labels={"estimated_count": "People", "zone": "Zone"},
            )
            fig.update_layout(height=350, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("📅 Match Schedule")
        if not matches.empty:
            display_cols = ["team_a", "team_b", "match_type", "status"]
            st.dataframe(
                matches[display_cols].head(5),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No matches scheduled.")


def _render_dashboard_alerts(overcrowded: list) -> None:
    """Render active crowd alerts."""
    if not overcrowded:
        return

    st.subheader("🚨 Active Alerts")
    for zone in overcrowded:
        st.markdown(
            f'<div class="alert-critical">'
            f'<b>⚠️ {zone["zone"]}</b> — {zone["density_level"]} '
            f'({zone["occupancy_percentage"]}% capacity)<br>'
            f'<small>Estimated: {zone["estimated_count"]:,} / '
            f'{zone["max_capacity"]:,}</small></div>',
            unsafe_allow_html=True,
        )


def render_fan_navigator(stadium_id: int, accessibility_mode: bool) -> None:
    """Render the fan navigation page with facility finder, directions, and map."""
    render_page_header("🧭", "Fan Navigator", "Find your way around the stadium with AI assistance")

    tab1, tab2, tab3 = st.tabs(["🔍 Find Facility", "🗺️ Get Directions", "🗺️ Stadium Map"])

    with tab1:
        _render_facility_finder(stadium_id, accessibility_mode)
    with tab2:
        _render_directions(stadium_id, accessibility_mode)
    with tab3:
        _render_stadium_map(stadium_id)


def _render_facility_finder(stadium_id: int, accessibility_mode: bool) -> None:
    """Render the facility finder interface."""
    st.subheader("Find Nearby Facilities")
    selected_facility = st.selectbox("What are you looking for?", FACILITY_TYPES)

    if st.button("🔍 Find", type="primary", key="find_facility"):
        facilities = find_facility(stadium_id, selected_facility, accessibility_mode)
        if facilities:
            st.success(f"Found {len(facilities)} {selected_facility}(s)")
            for facility in facilities[:5]:
                acc_badge = "♿ Accessible" if facility["is_accessible"] else ""
                st.markdown(
                    f"**{facility['icon']} {facility['type']}** — "
                    f"{facility['zone']}, Level {facility['floor_level']} {acc_badge}"
                )
        else:
            st.warning("No facilities found matching your criteria.")


def _render_directions(stadium_id: int, accessibility_mode: bool) -> None:
    """Render the AI-powered directions interface."""
    st.subheader("Get AI-Powered Directions")
    from_loc = st.text_input("Where are you now?", placeholder="e.g., Gate A, Section 101")
    to_loc = st.text_input("Where do you want to go?", placeholder="e.g., Nearest restroom")

    if st.button("🧭 Get Directions", type="primary", key="get_directions"):
        if not from_loc or not to_loc:
            st.warning("Please enter both starting point and destination.")
            return
        if not validate_api_key():
            return
        with st.spinner("Calculating best route..."):
            directions = get_navigation_directions(
                stadium_id, from_loc, to_loc, accessibility_mode
            )
        st.markdown("### 📍 Directions")
        st.markdown(directions)
        if accessibility_mode:
            st.info("♿ Route optimized for wheelchair accessibility")


def _render_stadium_map(stadium_id: int) -> None:
    """Render the interactive stadium map with facility markers."""
    st.subheader("Interactive Stadium Map")
    map_data = get_stadium_map_data(stadium_id)

    if not map_data:
        st.warning("Map data unavailable for this stadium.")
        return

    stadium_info = map_data["stadium"]
    stadium_map = folium.Map(
        location=[stadium_info["latitude"], stadium_info["longitude"]],
        zoom_start=16,
        tiles="CartoDB positron",
    )

    for facility in map_data["facilities"][:50]:
        color = "green" if facility["is_accessible"] else "blue"
        folium.CircleMarker(
            location=[facility["latitude"], facility["longitude"]],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            tooltip=f"{facility['icon']} {facility['type']} — {facility['zone']}",
            popup=facility["description"],
        ).add_to(stadium_map)

    st_folium(stadium_map, width=None, height=450, use_container_width=True)
    st.markdown("**Legend:** 🟢 Accessible | 🔵 Standard")


def render_crowd_monitor(stadium_id: int) -> None:
    """Render the crowd monitoring and prediction page."""
    render_page_header("👥", "Crowd Monitor", "Real-time crowd density and flow management")

    _render_crowd_density_chart(stadium_id)
    _render_overcrowded_zones(stadium_id)
    st.markdown("---")
    _render_crowd_prediction(stadium_id)


def _render_crowd_density_chart(stadium_id: int) -> None:
    """Render the current crowd density treemap."""
    st.subheader("📊 Current Zone Density")
    crowd_data = get_current_crowd_density(stadium_id)

    if crowd_data.empty:
        st.info("No crowd data available.")
        return

    fig = px.treemap(
        crowd_data,
        path=["zone"],
        values="estimated_count",
        color="density_level",
        color_discrete_map=DENSITY_COLORS,
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def _render_overcrowded_zones(stadium_id: int) -> None:
    """Render overcrowded zone metrics."""
    overcrowded = get_overcrowded_zones(stadium_id)
    if not overcrowded:
        return

    st.subheader("🚨 Overcrowded Zones")
    for zone in overcrowded:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Zone", zone["zone"])
        with col2:
            st.metric("Occupancy", f"{zone['occupancy_percentage']}%")
        with col3:
            st.metric("Level", zone["density_level"])


def _render_crowd_prediction(stadium_id: int) -> None:
    """Render AI crowd flow prediction."""
    st.subheader("🔮 AI Crowd Prediction")
    if st.button("Generate Crowd Flow Prediction", type="primary"):
        if not validate_api_key():
            return
        with st.spinner("Analyzing crowd patterns..."):
            prediction = predict_crowd_flow(stadium_id)
        st.markdown(prediction)


def render_multilingual_hub() -> None:
    """Render the multilingual translation and announcement page."""
    render_page_header("🌐", "Multilingual Hub", "Breaking language barriers for a global audience")

    languages = get_supported_languages()
    tab1, tab2 = st.tabs(["📝 Translate Text", "📢 Announcement Translator"])

    with tab1:
        _render_text_translator(languages)
    with tab2:
        _render_announcement_translator(languages)


def _render_text_translator(languages: dict) -> None:
    """Render the real-time text translation interface."""
    st.subheader("Real-Time Translation")
    col1, col2 = st.columns(2)

    with col1:
        source_lang = st.selectbox("From", list(languages.values()), index=0, key="src_lang")
        source_code = [k for k, v in languages.items() if v == source_lang][0]
        input_text = st.text_area(
            "Enter text", height=150, placeholder="Type or paste text here..."
        )

    with col2:
        target_lang = st.selectbox("To", list(languages.values()), index=1, key="tgt_lang")
        target_code = [k for k, v in languages.items() if v == target_lang][0]

        if st.button("🔄 Translate", type="primary"):
            if not input_text:
                st.warning("Please enter text to translate.")
                return
            if not validate_api_key():
                return
            with st.spinner("Translating..."):
                translated = translate_text(input_text, target_code, source_code)
            st.text_area("Translation", value=translated, height=150, disabled=True)

            if st.button("🔊 Listen", key="listen_translation"):
                try:
                    audio = generate_audio_announcement(translated, target_code)
                    st.audio(audio, format="audio/mp3")
                except Exception as exc:
                    logger.error("Audio generation failed: %s", exc)
                    st.error(f"Audio error: {exc}")


def _render_announcement_translator(languages: dict) -> None:
    """Render the broadcast announcement translator."""
    st.subheader("Broadcast Announcement in Multiple Languages")
    announcement = st.text_area(
        "Announcement (in English)",
        placeholder="e.g., Gate C is temporarily closed. Please use Gate D.",
        height=100,
    )
    target_langs = st.multiselect(
        "Translate to:",
        [v for k, v in languages.items() if k != "en"],
        default=["Spanish", "French", "Arabic"],
    )

    if st.button("📢 Generate Translations", type="primary"):
        if not announcement:
            st.warning("Please enter an announcement.")
            return
        if not validate_api_key():
            return
        target_codes = [k for k, v in languages.items() if v in target_langs]
        with st.spinner("Translating to all languages..."):
            translations = translate_announcement(announcement, target_codes)
        for lang_code, text in translations.items():
            lang_name = languages.get(lang_code, lang_code)
            st.markdown(f"**{lang_name}:** {text}")


def render_voice_assistant(stadium_id: int, accessibility_mode: bool) -> None:
    """Render the voice assistant page with STT and text query support."""
    render_page_header("🎤", "Voice Assistant", "Speak your questions — get instant AI answers")

    languages = get_supported_languages()
    language = st.selectbox(
        "Audio language",
        ["en", "es", "fr", "hi", "ar", "pt"],
        format_func=lambda x: languages.get(x, x),
    )

    _render_voice_upload(stadium_id, language)
    st.markdown("---")
    _render_text_query(stadium_id, language, accessibility_mode)


def _render_voice_upload(stadium_id: int, language: str) -> None:
    """Render the voice upload and processing interface."""
    st.markdown("### Upload a voice recording")
    audio_file = st.file_uploader(
        "Upload audio",
        type=["wav", "mp3", "m4a", "ogg", "webm"],
        help="Record a question about the stadium, facilities, or match",
    )

    if not audio_file:
        return

    st.audio(audio_file)

    if st.button("🎙️ Process Voice Query", type="primary"):
        if not validate_api_key():
            return
        with st.spinner("Processing your voice..."):
            result = process_voice_query(
                audio_file.getvalue(), stadium_id, language, audio_file.name
            )

        if "error" in result:
            st.error(result["error"])
        else:
            st.info(f"**You said:** {result['transcription']}")
            st.markdown("### 🤖 Response")
            st.markdown(result["response"])


def _render_text_query(stadium_id: int, language: str, accessibility_mode: bool) -> None:
    """Render the text-based query interface."""
    st.markdown("### 💬 Or type your question")
    text_query = st.text_input(
        "Ask anything about the stadium",
        placeholder="e.g., Where is the nearest accessible restroom?",
    )

    if st.button("Ask AI", key="text_ask") and text_query:
        if not validate_api_key():
            return
        with st.spinner("Thinking..."):
            result = process_fan_query(text_query, stadium_id, language, accessibility_mode)
        st.markdown(result["response"])


def render_smart_search(stadium_id: int) -> None:
    """Render the semantic search and NL-to-SQL query page."""
    render_page_header("🔍", "Smart Search", "Semantic search and natural language database queries")

    selected_stadium = STADIUMS_LIST[stadium_id - 1]
    tab1, tab2 = st.tabs(["🔎 Semantic Search", "🗄️ Database Query"])

    with tab1:
        _render_semantic_search(selected_stadium)
    with tab2:
        _render_database_query()


def _render_semantic_search(selected_stadium: str) -> None:
    """Render the semantic search interface."""
    search_query = st.text_input(
        "Search stadium knowledge base",
        placeholder="e.g., wheelchair accessible food options",
    )
    num_results = st.slider("Results", 3, 15, 8)

    if st.button("🔍 Search", type="primary") and search_query:
        with st.spinner("Searching..."):
            results = semantic_search(
                search_query, n_results=num_results, stadium_filter=selected_stadium
            )

        if results:
            st.success(f"Found {len(results)} results")
            for i, result in enumerate(results, 1):
                with st.expander(
                    f"#{i} | {result['metadata'].get('type', '')} | "
                    f"Relevance: {result['similarity']:.1%}"
                ):
                    st.markdown(result["document"])
                    st.caption(
                        f"Type: {result['metadata'].get('type', 'N/A')} | "
                        f"Stadium: {result['metadata'].get('stadium', 'N/A')}"
                    )
        else:
            st.info("No results found. Try a different query.")


def _render_database_query() -> None:
    """Render the natural language database query interface."""
    nl_query = st.text_input(
        "Ask in plain English",
        placeholder="e.g., Which stadiums have the most capacity?",
    )

    if st.button("🗄️ Query", type="primary") and nl_query:
        if not validate_api_key():
            return
        with st.spinner("Generating query..."):
            sql, results = query_database(nl_query)
        st.code(sql, language="sql")
        if not results.empty and "error" not in results.columns:
            st.dataframe(results, use_container_width=True, hide_index=True)
        else:
            st.warning("No results or query error.")


def render_operations_center(stadium_id: int) -> None:
    """Render the operations center for organizers and staff."""
    render_page_header(
        "📊", "Operations Center",
        "AI-powered operational intelligence for organizers and staff",
    )

    tab1, tab2, tab3 = st.tabs(["📈 Insights", "🚨 Emergency", "💬 Feedback"])

    with tab1:
        _render_operational_insights(stadium_id)
    with tab2:
        _render_emergency_response(stadium_id)
    with tab3:
        _render_feedback_analysis(stadium_id)


def _render_operational_insights(stadium_id: int) -> None:
    """Render AI-generated operational insights."""
    if st.button("🤖 Generate AI Insights", type="primary"):
        if not validate_api_key():
            return
        with st.spinner("Analyzing operations..."):
            insights = generate_operational_insights(stadium_id)

        st.markdown("### 🧠 AI Operational Insights")
        st.markdown(insights["insights"])

        st.markdown("---")
        st.markdown("### 📊 Feedback Summary")
        feedback = insights["feedback_summary"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Feedback", feedback.get("total_feedback", 0))
        with col2:
            sentiments = feedback.get("sentiments", {})
            positive = sentiments.get("Positive", 0)
            total = max(sum(sentiments.values()), 1)
            st.metric("Satisfaction", f"{(positive / total) * 100:.0f}%")


def _render_emergency_response(stadium_id: int) -> None:
    """Render the emergency response plan generator."""
    st.subheader("🚨 Emergency Response Generator")
    st.warning("⚠️ This tool generates AI-assisted response plans. Always follow official protocols.")

    emergency_type = st.selectbox("Emergency Type", EMERGENCY_TYPES)
    location = st.text_input(
        "Location in stadium", placeholder="e.g., Section 205, North Stand"
    )

    if st.button("🚨 Generate Response Plan", type="primary"):
        if not validate_api_key():
            return
        with st.spinner("Generating emergency response..."):
            response = generate_emergency_response(stadium_id, emergency_type, location)

        st.markdown("### 📋 Response Plan")
        st.markdown(response["response_plan"])

        st.markdown("### 📢 Multilingual Announcements")
        languages = get_supported_languages()
        for lang_code, text in response["announcements"].items():
            lang_name = languages.get(lang_code, lang_code)
            st.markdown(f"**{lang_name}:** {text}")


def _render_feedback_analysis(stadium_id: int) -> None:
    """Render fan feedback analysis with charts."""
    st.subheader("💬 Fan Feedback Analysis")
    feedback = get_feedback_summary(stadium_id)

    if not feedback:
        st.info("No feedback data available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        sentiments = feedback.get("sentiments", {})
        if sentiments:
            sent_df = pd.DataFrame(
                list(sentiments.items()), columns=["Sentiment", "Count"]
            )
            fig = px.pie(
                sent_df,
                values="Count",
                names="Sentiment",
                color="Sentiment",
                color_discrete_map=SENTIMENT_COLORS,
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        categories = feedback.get("top_categories", [])
        if categories:
            cat_df = pd.DataFrame(categories, columns=["Category", "Count"])
            fig = px.bar(cat_df, x="Category", y="Count", color="Category")
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# ==================== MAIN ====================


def main() -> None:
    """Main application entry point."""
    configure_page()
    initialize_app()

    page, stadium_id, accessibility_mode = render_sidebar()

    # Route to appropriate page renderer
    page_routes = {
        "🏠 Dashboard": lambda: render_dashboard(stadium_id),
        "🧭 Fan Navigator": lambda: render_fan_navigator(stadium_id, accessibility_mode),
        "👥 Crowd Monitor": lambda: render_crowd_monitor(stadium_id),
        "🌐 Multilingual Hub": render_multilingual_hub,
        "🎤 Voice Assistant": lambda: render_voice_assistant(
            stadium_id, accessibility_mode
        ),
        "🔍 Smart Search": lambda: render_smart_search(stadium_id),
        "📊 Operations Center": lambda: render_operations_center(stadium_id),
    }

    renderer = page_routes.get(page)
    if renderer:
        renderer()
    else:
        st.error("Page not found.")


if __name__ == "__main__":
    main()
