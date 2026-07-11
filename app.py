"""
StadiumAI — FIFA World Cup 2026 GenAI Stadium Operations Platform
Streamlit UI for fans, organizers, volunteers, and venue staff.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from data.seed_data import create_database, initialize_database
from tools.sql_tool import (
    get_match_schedule, get_feedback_summary, query_database
)
from tools.crowd_tool import (
    get_current_crowd_density, get_overcrowded_zones, predict_crowd_flow
)
from tools.navigation_tool import (
    find_facility, get_navigation_directions, get_stadium_map_data
)
from tools.multilingual_tool import (
    translate_text, get_supported_languages, generate_audio_announcement
)
from tools.rag_tool import index_stadium_data, semantic_search
from tools.voice_tool import speech_to_text, text_to_speech
from pipeline import (
    process_fan_query, process_voice_query,
    generate_operational_insights, generate_emergency_response
)

# --- Page Config ---
st.set_page_config(
    page_title="StadiumAI — FIFA World Cup 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Accessibility ---
st.markdown("""
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
    .metric-card {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #1565C0;
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
    /* Accessibility: Minimum touch target size */
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
    .skip-link:focus {
        top: 0;
    }
</style>
<a href="#main-content" class="skip-link" aria-label="Skip to main content">Skip to main content</a>
""", unsafe_allow_html=True)


# --- Initialize Database ---
@st.cache_resource
def init_db():
    """Initialize database and index on first run."""
    create_database()
    initialize_database()
    count = index_stadium_data()
    return count


init_db()


# --- Sidebar Navigation ---
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
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 🏟️ Select Stadium")
    
    stadiums_list = [
        "MetLife Stadium", "AT&T Stadium", "SoFi Stadium",
        "Estadio Azteca", "Hard Rock Stadium", "Lumen Field",
        "BMO Field", "NRG Stadium"
    ]
    selected_stadium = st.selectbox(
        "Stadium", stadiums_list, 
        label_visibility="collapsed"
    )
    stadium_id = stadiums_list.index(selected_stadium) + 1

    st.markdown("---")
    accessibility_mode = st.toggle("♿ Accessibility Mode", value=False,
                                    help="Enable to prioritize accessible routes and facilities")

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#666; font-size:0.8rem;'>
        Powered by<br>
        <b>Groq Llama 4 Scout</b><br>
        Whisper • Chroma • Streamlit
    </div>
    """, unsafe_allow_html=True)


# --- Helper Functions ---
def density_color(level: str) -> str:
    """Get color for crowd density level."""
    colors = {
        "Low": "#4CAF50",
        "Moderate": "#FFC107",
        "High": "#FF9800",
        "Very High": "#FF5722",
        "Critical": "#B71C1C"
    }
    return colors.get(level, "#9E9E9E")


# ==================== PAGES ====================

# --- Dashboard ---
if page == "🏠 Dashboard":
    st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)
    st.markdown('<p class="main-header">⚽ StadiumAI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Stadium Operations — FIFA World Cup 2026</p>', unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    crowd_data = get_current_crowd_density(stadium_id)
    overcrowded = get_overcrowded_zones(stadium_id)
    matches = get_match_schedule(stadium_id)
    feedback = get_feedback_summary(stadium_id)

    with col1:
        total_fans = crowd_data["estimated_count"].sum() if not crowd_data.empty else 0
        st.metric("👥 Current Fans", f"{total_fans:,}")
    with col2:
        st.metric("🚨 Crowded Zones", len(overcrowded))
    with col3:
        upcoming = len(matches[matches["status"] == "Scheduled"]) if not matches.empty else 0
        st.metric("📅 Upcoming Matches", upcoming)
    with col4:
        st.metric("💬 Fan Feedback", feedback.get("total_feedback", 0))

    st.markdown("---")

    # Crowd Overview & Match Schedule
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("👥 Crowd Density Overview")
        if not crowd_data.empty:
            fig = px.bar(
                crowd_data.head(10), x="zone", y="estimated_count",
                color="density_level",
                color_discrete_map={
                    "Low": "#4CAF50", "Moderate": "#FFC107",
                    "High": "#FF9800", "Very High": "#FF5722", "Critical": "#B71C1C"
                },
                labels={"estimated_count": "People", "zone": "Zone"}
            )
            fig.update_layout(height=350, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("📅 Match Schedule")
        if not matches.empty:
            display_matches = matches[["team_a", "team_b", "match_type", "status"]].head(5)
            st.dataframe(display_matches, use_container_width=True, hide_index=True)
        else:
            st.info("No matches scheduled.")

    # Alerts
    if overcrowded:
        st.subheader("🚨 Active Alerts")
        for zone in overcrowded:
            st.markdown(f"""
            <div class="alert-critical">
                <b>⚠️ {zone['zone']}</b> — {zone['density_level']} 
                ({zone['occupancy_percentage']}% capacity)<br>
                <small>Estimated: {zone['estimated_count']:,} / {zone['max_capacity']:,}</small>
            </div>
            """, unsafe_allow_html=True)


# --- Fan Navigator ---
elif page == "🧭 Fan Navigator":
    st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)
    st.markdown('<p class="main-header">🧭 Fan Navigator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Find your way around the stadium with AI assistance</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔍 Find Facility", "🗺️ Get Directions", "🗺️ Stadium Map"])

    with tab1:
        st.subheader("Find Nearby Facilities")
        facility_types = [
            "Restroom", "Food Court", "First Aid", "Information Desk",
            "Merchandise Store", "ATM", "Prayer Room", "Baby Care Room",
            "Wheelchair Access Point", "Water Station", "Charging Station",
            "Exit Gate"
        ]
        selected_facility = st.selectbox("What are you looking for?", facility_types)

        if st.button("🔍 Find", type="primary", key="find_facility"):
            facilities = find_facility(stadium_id, selected_facility, accessibility_mode)
            if facilities:
                st.success(f"Found {len(facilities)} {selected_facility}(s)")
                for f in facilities[:5]:
                    acc_badge = "♿ Accessible" if f["is_accessible"] else ""
                    st.markdown(f"""
                    **{f['icon']} {f['type']}** — {f['zone']}, Level {f['floor_level']} {acc_badge}
                    """)
            else:
                st.warning("No facilities found matching your criteria.")

    with tab2:
        st.subheader("Get AI-Powered Directions")
        from_loc = st.text_input("Where are you now?", placeholder="e.g., Gate A, Section 101")
        to_loc = st.text_input("Where do you want to go?", placeholder="e.g., Nearest restroom, Food Court")

        if st.button("🧭 Get Directions", type="primary", key="get_directions"):
            if from_loc and to_loc:
                if not os.getenv("GROQ_API_KEY"):
                    st.error("GROQ_API_KEY not set!")
                else:
                    with st.spinner("Calculating best route..."):
                        directions = get_navigation_directions(
                            stadium_id, from_loc, to_loc, accessibility_mode
                        )
                    st.markdown("### 📍 Directions")
                    st.markdown(directions)

                    if accessibility_mode:
                        st.info("♿ Route optimized for wheelchair accessibility")
            else:
                st.warning("Please enter both starting point and destination.")

    with tab3:
        st.subheader("Interactive Stadium Map")
        map_data = get_stadium_map_data(stadium_id)

        if map_data:
            stadium_info = map_data["stadium"]
            m = folium.Map(
                location=[stadium_info["latitude"], stadium_info["longitude"]],
                zoom_start=16, tiles="CartoDB positron"
            )

            for f in map_data["facilities"][:50]:
                color = "green" if f["is_accessible"] else "blue"
                folium.CircleMarker(
                    location=[f["latitude"], f["longitude"]],
                    radius=5, color=color, fill=True,
                    fill_opacity=0.7,
                    tooltip=f"{f['icon']} {f['type']} — {f['zone']}",
                    popup=f"{f['description']}"
                ).add_to(m)

            st_folium(m, width=None, height=450, use_container_width=True)

            st.markdown("**Legend:** 🟢 Accessible | 🔵 Standard")


# --- Crowd Monitor ---
elif page == "👥 Crowd Monitor":
    st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)
    st.markdown('<p class="main-header">👥 Crowd Monitor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time crowd density and flow management</p>', unsafe_allow_html=True)

    # Current density
    st.subheader("📊 Current Zone Density")
    crowd_data = get_current_crowd_density(stadium_id)

    if not crowd_data.empty:
        # Heatmap-style display
        fig = px.treemap(
            crowd_data, path=["zone"], values="estimated_count",
            color="density_level",
            color_discrete_map={
                "Low": "#4CAF50", "Moderate": "#FFC107",
                "High": "#FF9800", "Very High": "#FF5722", "Critical": "#B71C1C"
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Overcrowded zones
    overcrowded = get_overcrowded_zones(stadium_id)
    if overcrowded:
        st.subheader("🚨 Overcrowded Zones")
        for zone in overcrowded:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Zone", zone["zone"])
            with col2:
                st.metric("Occupancy", f"{zone['occupancy_percentage']}%")
            with col3:
                st.metric("Level", zone["density_level"])

    st.markdown("---")

    # AI Prediction
    st.subheader("🔮 AI Crowd Prediction")
    if st.button("Generate Crowd Flow Prediction", type="primary"):
        if not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY not set!")
        else:
            with st.spinner("Analyzing crowd patterns..."):
                prediction = predict_crowd_flow(stadium_id)
            st.markdown(prediction)


# --- Multilingual Hub ---
elif page == "🌐 Multilingual Hub":
    st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)
    st.markdown('<p class="main-header">🌐 Multilingual Hub</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Breaking language barriers for a global audience</p>', unsafe_allow_html=True)

    languages = get_supported_languages()

    tab1, tab2 = st.tabs(["📝 Translate Text", "📢 Announcement Translator"])

    with tab1:
        st.subheader("Real-Time Translation")
        col1, col2 = st.columns(2)

        with col1:
            source_lang = st.selectbox(
                "From", list(languages.values()), index=0, key="src_lang"
            )
            source_code = [k for k, v in languages.items() if v == source_lang][0]
            input_text = st.text_area("Enter text", height=150,
                                       placeholder="Type or paste text here...")

        with col2:
            target_lang = st.selectbox(
                "To", list(languages.values()), index=1, key="tgt_lang"
            )
            target_code = [k for k, v in languages.items() if v == target_lang][0]

            if st.button("🔄 Translate", type="primary"):
                if input_text and os.getenv("GROQ_API_KEY"):
                    with st.spinner("Translating..."):
                        translated = translate_text(input_text, target_code, source_code)
                    st.text_area("Translation", value=translated, height=150, disabled=True)

                    # Audio option
                    if st.button("🔊 Listen", key="listen_translation"):
                        try:
                            audio = generate_audio_announcement(translated, target_code)
                            st.audio(audio, format="audio/mp3")
                        except Exception as e:
                            st.error(f"Audio error: {e}")
                elif not os.getenv("GROQ_API_KEY"):
                    st.error("GROQ_API_KEY not set!")

    with tab2:
        st.subheader("Broadcast Announcement in Multiple Languages")
        announcement = st.text_area(
            "Announcement (in English)",
            placeholder="e.g., Gate C is temporarily closed. Please use Gate D.",
            height=100
        )
        target_langs = st.multiselect(
            "Translate to:", 
            [v for k, v in languages.items() if k != "en"],
            default=["Spanish", "French", "Arabic"]
        )

        if st.button("📢 Generate Translations", type="primary"):
            if announcement and os.getenv("GROQ_API_KEY"):
                target_codes = [k for k, v in languages.items() if v in target_langs]
                with st.spinner("Translating to all languages..."):
                    from tools.multilingual_tool import translate_announcement
                    translations = translate_announcement(announcement, target_codes)

                for lang_code, text in translations.items():
                    lang_name = languages.get(lang_code, lang_code)
                    st.markdown(f"**{lang_name}:** {text}")


# --- Voice Assistant ---
elif page == "🎤 Voice Assistant":
    st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)
    st.markdown('<p class="main-header">🎤 Voice Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Speak your questions — get instant AI answers</p>', unsafe_allow_html=True)

    st.markdown("### Upload a voice recording")
    audio_file = st.file_uploader(
        "Upload audio", type=["wav", "mp3", "m4a", "ogg", "webm"],
        help="Record a question about the stadium, facilities, or match"
    )

    language = st.selectbox(
        "Audio language", 
        ["en", "es", "fr", "hi", "ar", "pt"],
        format_func=lambda x: get_supported_languages().get(x, x)
    )

    if audio_file:
        st.audio(audio_file)

        if st.button("🎙️ Process Voice Query", type="primary"):
            if not os.getenv("GROQ_API_KEY"):
                st.error("GROQ_API_KEY not set!")
            else:
                with st.spinner("Processing your voice..."):
                    result = process_voice_query(
                        audio_file.getvalue(), stadium_id,
                        language, audio_file.name
                    )

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.info(f"**You said:** {result['transcription']}")
                    st.markdown("### 🤖 Response")
                    st.markdown(result["response"])

                    # Read response aloud
                    if st.button("🔊 Read Response"):
                        try:
                            audio_bytes = text_to_speech(result["response"][:500], language)
                            st.audio(audio_bytes, format="audio/mp3")
                        except Exception as e:
                            st.error(f"Audio error: {e}")

    st.markdown("---")
    st.markdown("### 💬 Or type your question")
    text_query = st.text_input(
        "Ask anything about the stadium",
        placeholder="e.g., Where is the nearest accessible restroom?"
    )

    if st.button("Ask AI", key="text_ask") and text_query:
        if not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY not set!")
        else:
            with st.spinner("Thinking..."):
                result = process_fan_query(
                    text_query, stadium_id, language, accessibility_mode
                )
            st.markdown(result["response"])


# --- Smart Search ---
elif page == "🔍 Smart Search":
    st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)
    st.markdown('<p class="main-header">🔍 Smart Search</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Semantic search and natural language database queries</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔎 Semantic Search", "🗄️ Database Query"])

    with tab1:
        search_query = st.text_input(
            "Search stadium knowledge base",
            placeholder="e.g., wheelchair accessible food options, prayer room location"
        )
        num_results = st.slider("Results", 3, 15, 8)

        if st.button("🔍 Search", type="primary") and search_query:
            with st.spinner("Searching..."):
                results = semantic_search(
                    search_query, n_results=num_results,
                    stadium_filter=selected_stadium
                )

            if results:
                st.success(f"Found {len(results)} results")
                for i, r in enumerate(results, 1):
                    with st.expander(
                        f"#{i} | {r['metadata'].get('type', '')} | "
                        f"Relevance: {r['similarity']:.1%}"
                    ):
                        st.markdown(r["document"])
                        st.caption(f"Type: {r['metadata'].get('type', 'N/A')} | "
                                   f"Stadium: {r['metadata'].get('stadium', 'N/A')}")

    with tab2:
        nl_query = st.text_input(
            "Ask in plain English",
            placeholder="e.g., Which stadiums have the most capacity?"
        )

        if st.button("🗄️ Query", type="primary") and nl_query:
            if not os.getenv("GROQ_API_KEY"):
                st.error("GROQ_API_KEY not set!")
            else:
                with st.spinner("Generating query..."):
                    sql, results = query_database(nl_query)
                st.code(sql, language="sql")
                if not results.empty and "error" not in results.columns:
                    st.dataframe(results, use_container_width=True, hide_index=True)
                else:
                    st.warning("No results or query error.")


# --- Operations Center ---
elif page == "📊 Operations Center":
    st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)
    st.markdown('<p class="main-header">📊 Operations Center</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered operational intelligence for organizers and staff</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Insights", "🚨 Emergency", "💬 Feedback"])

    with tab1:
        if st.button("🤖 Generate AI Insights", type="primary"):
            if not os.getenv("GROQ_API_KEY"):
                st.error("GROQ_API_KEY not set!")
            else:
                with st.spinner("Analyzing operations..."):
                    insights = generate_operational_insights(stadium_id)

                st.markdown("### 🧠 AI Operational Insights")
                st.markdown(insights["insights"])

                st.markdown("---")
                st.markdown("### 📊 Feedback Summary")
                fb = insights["feedback_summary"]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Feedback", fb.get("total_feedback", 0))
                with col2:
                    sentiments = fb.get("sentiments", {})
                    positive = sentiments.get("Positive", 0)
                    total = sum(sentiments.values()) if sentiments else 1
                    st.metric("Satisfaction", f"{(positive/total)*100:.0f}%")

    with tab2:
        st.subheader("🚨 Emergency Response Generator")
        st.warning("⚠️ This tool generates AI-assisted response plans. Always follow official protocols.")

        emergency_type = st.selectbox(
            "Emergency Type",
            ["Medical Emergency", "Fire", "Structural Issue", "Crowd Crush",
             "Severe Weather", "Security Threat", "Power Failure"]
        )
        location = st.text_input("Location in stadium", placeholder="e.g., Section 205, North Stand")

        if st.button("🚨 Generate Response Plan", type="primary"):
            if not os.getenv("GROQ_API_KEY"):
                st.error("GROQ_API_KEY not set!")
            else:
                with st.spinner("Generating emergency response..."):
                    response = generate_emergency_response(
                        stadium_id, emergency_type, location
                    )

                st.markdown("### 📋 Response Plan")
                st.markdown(response["response_plan"])

                st.markdown("### 📢 Multilingual Announcements")
                for lang, text in response["announcements"].items():
                    lang_name = get_supported_languages().get(lang, lang)
                    st.markdown(f"**{lang_name}:** {text}")

    with tab3:
        st.subheader("💬 Fan Feedback Analysis")
        feedback = get_feedback_summary(stadium_id)

        if feedback:
            col1, col2 = st.columns(2)
            with col1:
                sentiments = feedback.get("sentiments", {})
                if sentiments:
                    sent_df = pd.DataFrame(
                        list(sentiments.items()), columns=["Sentiment", "Count"]
                    )
                    fig = px.pie(sent_df, values="Count", names="Sentiment",
                                 color="Sentiment",
                                 color_discrete_map={
                                     "Positive": "#4CAF50",
                                     "Neutral": "#FFC107",
                                     "Negative": "#F44336"
                                 })
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                categories = feedback.get("top_categories", [])
                if categories:
                    cat_df = pd.DataFrame(categories, columns=["Category", "Count"])
                    fig = px.bar(cat_df, x="Category", y="Count", color="Category")
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
