"""
Voice Tool — Speech-to-text (Groq Whisper) and text-to-speech (gTTS).
Enables voice-based interaction for fans and staff.
"""

import os
import tempfile
from typing import Optional
from groq import Groq
from gtts import gTTS


def get_client() -> Groq:
    """Get Groq client with API key from environment."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return Groq(api_key=api_key)


def speech_to_text(audio_bytes: bytes, filename: str = "recording.wav",
                   language: str = "en") -> str:
    """
    Transcribe audio using Groq Whisper Large V3.
    
    Args:
        audio_bytes: Raw audio file bytes
        filename: Original filename for format detection
        language: Expected language of the audio
    
    Returns:
        Transcribed text string
    """
    if not audio_bytes:
        return ""

    client = get_client()

    try:
        suffix = os.path.splitext(filename)[1] if "." in filename else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                language=language,
                response_format="text"
            )

        os.unlink(tmp_path)

        if isinstance(transcription, str):
            return transcription.strip()
        return transcription.text.strip()

    except Exception as e:
        return f"Transcription error: {str(e)}"


def text_to_speech(text: str, language: str = "en") -> bytes:
    """
    Convert text to speech using gTTS.
    
    Args:
        text: Text to convert to speech
        language: Language code for speech output
    
    Returns:
        Audio bytes in MP3 format
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty for speech generation.")

    # Map language codes to gTTS-compatible codes
    lang_map = {
        "en": "en", "es": "es", "fr": "fr", "ar": "ar",
        "hi": "hi", "pt": "pt", "de": "de", "ja": "ja",
        "ko": "ko", "zh": "zh-CN", "it": "it",
    }
    gtts_lang = lang_map.get(language, "en")

    try:
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tts.save(tmp.name)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()

        os.unlink(tmp_path)
        return audio_bytes

    except Exception as e:
        raise Exception(f"Speech generation error: {str(e)}")


def generate_emergency_alert(message: str, language: str = "en") -> bytes:
    """
    Generate an emergency audio alert with urgency prefix.
    
    Args:
        message: Alert message content
        language: Language for the alert
    
    Returns:
        Audio bytes of the emergency alert
    """
    prefixes = {
        "en": "Attention! Emergency Alert!",
        "es": "¡Atención! ¡Alerta de emergencia!",
        "fr": "Attention! Alerte d'urgence!",
        "ar": "انتباه! تنبيه طوارئ!",
        "hi": "ध्यान दें! आपातकालीन चेतावनी!",
    }
    prefix = prefixes.get(language, prefixes["en"])
    full_message = f"{prefix} {message}"
    return text_to_speech(full_message, language)
