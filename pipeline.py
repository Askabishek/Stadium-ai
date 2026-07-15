"""
Core Pipeline — Single multimodal pipeline for StadiumAI.

Orchestrates navigation, crowd management, multilingual, voice, and RAG tools
into a unified processing flow for fan queries, voice input, operational
insights, and emergency response.

Author: Abishek
License: MIT
"""

import os
import logging
from typing import Optional

from groq import Groq

from tools.navigation_tool import (
    find_facility,
    get_navigation_directions,
    get_nearest_exit,
    get_stadium_map_data,
)
from tools.crowd_tool import (
    get_current_crowd_density,
    get_overcrowded_zones,
    predict_crowd_flow,
    suggest_alternate_routes,
)
from tools.multilingual_tool import (
    translate_text,
    translate_announcement,
    detect_language,
    multilingual_chat_response,
    get_supported_languages,
)
from tools.voice_tool import speech_to_text, text_to_speech
from tools.rag_tool import semantic_search, index_stadium_data
from tools.sql_tool import query_database, get_match_schedule, get_feedback_summary

logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_RESPONSE_TOKENS = 768
MAX_EMERGENCY_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.3
EMERGENCY_TEMPERATURE = 0.2
RAG_RESULT_COUNT = 5
TTS_MAX_CHARS = 500
EMERGENCY_LANGUAGES = ["es", "fr", "ar", "hi", "pt"]


def _get_client() -> Groq:
    """
    Get Groq client with API key from environment.

    Returns:
        Configured Groq client instance.

    Raises:
        ValueError: If GROQ_API_KEY environment variable is not set.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return Groq(api_key=api_key)


def _generate_llm_response(prompt: str, max_tokens: int = MAX_RESPONSE_TOKENS,
                           temperature: float = DEFAULT_TEMPERATURE) -> str:
    """
    Generate a response from the LLM with standardized error handling.

    Args:
        prompt: The prompt to send to the LLM.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature for generation.

    Returns:
        Generated text response or error message.
    """
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        return f"I apologize, I'm having trouble processing your request: {str(exc)}"


def process_fan_query(
    query: str,
    stadium_id: int = 1,
    language: str = "en",
    accessibility_needed: bool = False,
) -> dict:
    """
    Process a fan's query using the full AI pipeline.

    Pipeline: Query -> Language Detection -> RAG Search -> LLM Response -> Translation

    Args:
        query: Fan's question or request in any supported language.
        stadium_id: ID of the current stadium (1-8).
        language: Preferred response language code or 'auto' for detection.
        accessibility_needed: Whether to prioritize accessibility information.

    Returns:
        Dictionary containing:
            - response: Final response text in the user's language.
            - original_language: Detected language code.
            - sources: Top RAG search results used as context.
            - accessibility_info_included: Whether accessibility was prioritized.
    """
    logger.info("Processing fan query: '%s' (stadium=%d, lang=%s)", query, stadium_id, language)

    # Step 1: Detect language if set to auto
    detected_lang = detect_language(query) if language == "auto" else language

    # Step 2: Translate query to English for processing if needed
    processing_query = query
    if detected_lang != "en":
        processing_query = translate_text(query, "en", detected_lang)

    # Step 3: Search knowledge base for relevant context
    rag_results = semantic_search(processing_query, n_results=RAG_RESULT_COUNT)
    context = "\n".join([r["document"] for r in rag_results])

    # Step 4: Build prompt and generate response
    accessibility_note = (
        "The user requires accessibility information. Prioritize wheelchair-accessible "
        "options, ramps, elevators, and accessible facilities."
        if accessibility_needed else ""
    )

    prompt = (
        "You are StadiumAI, an intelligent assistant for FIFA World Cup 2026.\n"
        "Help fans, staff, and organizers with stadium operations.\n\n"
        f"Context from knowledge base:\n{context}\n\n"
        f"{accessibility_note}\n\n"
        f"User query: {processing_query}\n\n"
        "Provide a helpful, concise, and accurate response. "
        "Include specific locations, directions, or data when available. "
        "If suggesting facilities, mention accessibility status."
    )

    answer = _generate_llm_response(prompt)

    # Step 5: Translate response back to user's language if needed
    final_response = answer
    if detected_lang != "en":
        final_response = translate_text(answer, detected_lang, "en")

    return {
        "response": final_response,
        "original_language": detected_lang,
        "sources": rag_results[:3],
        "accessibility_info_included": accessibility_needed,
    }


def process_voice_query(
    audio_bytes: bytes,
    stadium_id: int = 1,
    language: str = "en",
    filename: str = "recording.wav",
) -> dict:
    """
    Process a voice query from a fan.

    Pipeline: Audio -> Whisper STT -> Process Query -> Generate Response

    Args:
        audio_bytes: Raw audio file bytes from the recording.
        stadium_id: ID of the current stadium (1-8).
        language: Expected language of the audio recording.
        filename: Original filename for format detection.

    Returns:
        Dictionary containing:
            - transcription: Transcribed text from audio.
            - response: AI-generated response to the query.
            - sources: RAG sources used for context.
            - language: Detected/specified language code.
            - error (optional): Error message if transcription failed.
    """
    logger.info("Processing voice query (stadium=%d, lang=%s)", stadium_id, language)

    # Step 1: Transcribe audio using Whisper
    transcription = speech_to_text(audio_bytes, filename, language)

    if transcription.startswith("Transcription error"):
        logger.warning("Transcription failed: %s", transcription)
        return {"error": transcription, "transcription": ""}

    # Step 2: Process the transcribed query through the text pipeline
    result = process_fan_query(transcription, stadium_id, language)

    return {
        "transcription": transcription,
        "response": result["response"],
        "sources": result.get("sources", []),
        "language": result.get("original_language", language),
    }


def generate_operational_insights(stadium_id: int) -> dict:
    """
    Generate AI-powered operational insights for organizers.

    Combines crowd predictions, zone alerts, and feedback data to produce
    actionable recommendations for stadium operations staff.

    Args:
        stadium_id: ID of the stadium to analyze (1-8).

    Returns:
        Dictionary containing:
            - insights: AI-generated operational recommendations.
            - crowd_prediction: Crowd flow prediction text.
            - overcrowded_zones: List of zones at/near capacity.
            - feedback_summary: Aggregated fan feedback statistics.
    """
    logger.info("Generating operational insights for stadium %d", stadium_id)

    # Gather operational data
    crowd_prediction = predict_crowd_flow(stadium_id)
    overcrowded = get_overcrowded_zones(stadium_id)
    feedback = get_feedback_summary(stadium_id)

    # Generate comprehensive insights
    overcrowded_info = overcrowded if overcrowded else "None currently"

    prompt = (
        "As an AI operations manager for FIFA World Cup 2026, "
        "provide a brief operational summary.\n\n"
        f"Crowd Prediction: {crowd_prediction}\n\n"
        f"Overcrowded Zones: {overcrowded_info}\n\n"
        f"Fan Feedback Summary: {feedback}\n\n"
        "Provide:\n"
        "1. TOP 3 immediate actions needed\n"
        "2. Resource deployment recommendations\n"
        "3. Potential issues to watch for in next 2 hours\n"
        "4. Sustainability tip for current operations\n\n"
        "Be concise and actionable."
    )

    insights = _generate_llm_response(prompt)

    return {
        "insights": insights,
        "crowd_prediction": crowd_prediction,
        "overcrowded_zones": overcrowded,
        "feedback_summary": feedback,
    }


def generate_emergency_response(
    stadium_id: int,
    emergency_type: str,
    location: str = "",
) -> dict:
    """
    Generate AI-assisted emergency response plan with multilingual alerts.

    Produces an immediate action plan and translates emergency announcements
    into multiple languages for international audiences.

    Args:
        stadium_id: ID of the stadium where emergency is occurring (1-8).
        emergency_type: Type of emergency (e.g., 'Fire', 'Medical Emergency').
        location: Specific location within the stadium, if known.

    Returns:
        Dictionary containing:
            - response_plan: Detailed emergency response plan text.
            - announcements: Dict mapping language codes to translated alerts.
            - emergency_type: Echo of the emergency type.
            - location: Echo of the location.
    """
    logger.warning(
        "Emergency response requested: type=%s, location=%s, stadium=%d",
        emergency_type, location, stadium_id,
    )

    location_info = location if location else "Unknown"

    prompt = (
        "EMERGENCY RESPONSE PLAN NEEDED.\n\n"
        f"Emergency Type: {emergency_type}\n"
        f"Location: {location_info}\n"
        f"Stadium ID: {stadium_id}\n\n"
        "Generate an immediate response plan including:\n"
        "1. IMMEDIATE ACTIONS (first 60 seconds)\n"
        "2. EVACUATION ROUTES (if needed)\n"
        "3. STAFF DEPLOYMENT\n"
        "4. PUBLIC ANNOUNCEMENT (clear, calm, informative)\n"
        "5. EXTERNAL SERVICES TO CONTACT\n\n"
        "Keep the announcement suitable for a diverse, international crowd."
    )

    plan = _generate_llm_response(prompt, max_tokens=MAX_EMERGENCY_TOKENS,
                                  temperature=EMERGENCY_TEMPERATURE)

    # Generate multilingual emergency announcements
    announcement = (
        f"Emergency alert: {emergency_type}. "
        "Please follow staff instructions and proceed to nearest exit calmly."
    )
    translations = translate_announcement(announcement, EMERGENCY_LANGUAGES)

    return {
        "response_plan": plan,
        "announcements": translations,
        "emergency_type": emergency_type,
        "location": location,
    }
