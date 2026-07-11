"""
Core Pipeline — Single multimodal pipeline for StadiumAI.
Orchestrates navigation, crowd management, multilingual, voice, and RAG tools.
"""

import os
from typing import Optional
from groq import Groq
from tools.navigation_tool import (
    find_facility, get_navigation_directions, get_nearest_exit, get_stadium_map_data
)
from tools.crowd_tool import (
    get_current_crowd_density, get_overcrowded_zones,
    predict_crowd_flow, suggest_alternate_routes
)
from tools.multilingual_tool import (
    translate_text, translate_announcement, detect_language,
    multilingual_chat_response, get_supported_languages
)
from tools.voice_tool import speech_to_text, text_to_speech
from tools.rag_tool import semantic_search, index_stadium_data
from tools.sql_tool import query_database, get_match_schedule, get_feedback_summary


def get_client() -> Groq:
    """Get Groq client with API key from environment."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return Groq(api_key=api_key)


def process_fan_query(query: str, stadium_id: int = 1,
                      language: str = "en",
                      accessibility_needed: bool = False) -> dict:
    """
    Process a fan's query using the full AI pipeline.
    
    Pipeline: Query → Language Detection → RAG Search → LLM Response → Translation
    
    Args:
        query: Fan's question or request
        stadium_id: Current stadium ID
        language: Preferred response language
        accessibility_needed: Whether accessibility info is needed
    
    Returns:
        Dictionary with response, sources, and metadata
    """
    # Step 1: Detect language if not English
    detected_lang = detect_language(query) if language == "auto" else language

    # Step 2: If not English, translate query for processing
    processing_query = query
    if detected_lang != "en":
        processing_query = translate_text(query, "en", detected_lang)

    # Step 3: Search knowledge base
    rag_results = semantic_search(processing_query, n_results=5)
    context = "\n".join([r["document"] for r in rag_results])

    # Step 4: Generate response
    client = get_client()

    accessibility_note = ""
    if accessibility_needed:
        accessibility_note = "The user requires accessibility information. Prioritize wheelchair-accessible options, ramps, elevators, and accessible facilities."

    prompt = f"""You are StadiumAI, an intelligent assistant for FIFA World Cup 2026.
Help fans, staff, and organizers with stadium operations.

Context from knowledge base:
{context}

{accessibility_note}

User query: {processing_query}

Provide a helpful, concise, and accurate response. 
Include specific locations, directions, or data when available.
If suggesting facilities, mention accessibility status."""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=768,
            temperature=0.3
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"I apologize, I'm having trouble processing your request: {str(e)}"

    # Step 5: Translate response if needed
    final_response = answer
    if detected_lang != "en":
        final_response = translate_text(answer, detected_lang, "en")

    return {
        "response": final_response,
        "original_language": detected_lang,
        "sources": rag_results[:3],
        "accessibility_info_included": accessibility_needed
    }


def process_voice_query(audio_bytes: bytes, stadium_id: int = 1,
                        language: str = "en",
                        filename: str = "recording.wav") -> dict:
    """
    Process a voice query from a fan.
    
    Pipeline: Audio → Whisper STT → Process Query → Generate Response
    
    Args:
        audio_bytes: Raw audio bytes
        stadium_id: Current stadium ID
        language: Expected audio language
        filename: Audio filename
    
    Returns:
        Dictionary with transcription and response
    """
    # Step 1: Transcribe audio
    transcription = speech_to_text(audio_bytes, filename, language)

    if transcription.startswith("Transcription error"):
        return {"error": transcription, "transcription": ""}

    # Step 2: Process the transcribed query
    result = process_fan_query(transcription, stadium_id, language)

    return {
        "transcription": transcription,
        "response": result["response"],
        "sources": result.get("sources", []),
        "language": result.get("original_language", language)
    }


def generate_operational_insights(stadium_id: int) -> dict:
    """
    Generate AI-powered operational insights for organizers.
    
    Args:
        stadium_id: Stadium to analyze
    
    Returns:
        Dictionary with crowd predictions, alerts, and recommendations
    """
    # Get crowd data
    crowd_prediction = predict_crowd_flow(stadium_id)
    overcrowded = get_overcrowded_zones(stadium_id)
    feedback = get_feedback_summary(stadium_id)

    # Generate overall insights
    client = get_client()

    prompt = f"""As an AI operations manager for FIFA World Cup 2026, provide a brief operational summary.

Crowd Prediction: {crowd_prediction}

Overcrowded Zones: {overcrowded if overcrowded else 'None currently'}

Fan Feedback Summary: {feedback}

Provide:
1. TOP 3 immediate actions needed
2. Resource deployment recommendations
3. Potential issues to watch for in next 2 hours
4. Sustainability tip for current operations

Be concise and actionable."""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=768,
            temperature=0.3
        )
        insights = response.choices[0].message.content
    except Exception as e:
        insights = f"Error generating insights: {str(e)}"

    return {
        "insights": insights,
        "crowd_prediction": crowd_prediction,
        "overcrowded_zones": overcrowded,
        "feedback_summary": feedback
    }


def generate_emergency_response(stadium_id: int, 
                                emergency_type: str,
                                location: str = "") -> dict:
    """
    Generate AI-assisted emergency response plan.
    
    Args:
        stadium_id: Stadium where emergency is occurring
        emergency_type: Type of emergency
        location: Location within stadium
    
    Returns:
        Emergency response plan with multilingual announcements
    """
    client = get_client()

    prompt = f"""EMERGENCY RESPONSE PLAN NEEDED.

Emergency Type: {emergency_type}
Location: {location if location else 'Unknown'}
Stadium ID: {stadium_id}

Generate an immediate response plan including:
1. IMMEDIATE ACTIONS (first 60 seconds)
2. EVACUATION ROUTES (if needed)
3. STAFF DEPLOYMENT
4. PUBLIC ANNOUNCEMENT (clear, calm, informative)
5. EXTERNAL SERVICES TO CONTACT

Keep the announcement suitable for a diverse, international crowd."""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.2
        )
        plan = response.choices[0].message.content
    except Exception as e:
        plan = f"Error: {str(e)}. Follow standard emergency protocols."

    # Generate multilingual announcements
    announcement = f"Emergency alert: {emergency_type}. Please follow staff instructions and proceed to nearest exit calmly."
    translations = translate_announcement(
        announcement, ["es", "fr", "ar", "hi", "pt"]
    )

    return {
        "response_plan": plan,
        "announcements": translations,
        "emergency_type": emergency_type,
        "location": location
    }
