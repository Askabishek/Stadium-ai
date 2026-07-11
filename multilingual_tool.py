"""
Multilingual Tool — Translation and multilingual assistance for global fans.
Supports real-time translation of announcements, signage, and conversations.
"""

import os
from typing import Optional
from groq import Groq
from gtts import gTTS
import tempfile

SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "ar": "Arabic",
    "hi": "Hindi",
    "pt": "Portuguese",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Mandarin Chinese",
    "it": "Italian",
    "nl": "Dutch",
    "ru": "Russian",
    "tr": "Turkish",
}

# gTTS supported language codes
GTTS_LANG_MAP = {
    "en": "en", "es": "es", "fr": "fr", "ar": "ar",
    "hi": "hi", "pt": "pt", "de": "de", "ja": "ja",
    "ko": "ko", "zh": "zh-CN", "it": "it", "nl": "nl",
    "ru": "ru", "tr": "tr",
}


def get_client() -> Groq:
    """Get Groq client with API key from environment."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return Groq(api_key=api_key)


def translate_text(text: str, target_language: str, 
                   source_language: str = "en") -> str:
    """
    Translate text to target language using Groq LLM.
    
    Args:
        text: Text to translate
        target_language: Target language code (e.g., 'es', 'fr')
        source_language: Source language code (default: 'en')
    
    Returns:
        Translated text
    """
    if not text or not text.strip():
        return ""

    target_name = SUPPORTED_LANGUAGES.get(target_language, target_language)
    source_name = SUPPORTED_LANGUAGES.get(source_language, source_language)

    if target_language == source_language:
        return text

    client = get_client()

    prompt = f"""Translate the following text from {source_name} to {target_name}.
Return ONLY the translated text, nothing else.

Text: {text}

Translation:"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Translation error: {str(e)}"


def translate_announcement(announcement: str, target_languages: list) -> dict:
    """
    Translate a stadium announcement into multiple languages.
    
    Args:
        announcement: Original announcement text
        target_languages: List of language codes to translate to
    
    Returns:
        Dictionary mapping language codes to translated text
    """
    translations = {"en": announcement}

    for lang_code in target_languages:
        if lang_code != "en":
            translations[lang_code] = translate_text(announcement, lang_code)

    return translations


def generate_audio_announcement(text: str, language: str = "en") -> bytes:
    """
    Generate audio from text in the specified language using gTTS.
    
    Args:
        text: Text to convert to speech
        language: Language code for speech generation
    
    Returns:
        Audio bytes in MP3 format
    """
    gtts_lang = GTTS_LANG_MAP.get(language, "en")

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
        raise Exception(f"Audio generation error: {str(e)}")


def detect_language(text: str) -> str:
    """
    Detect the language of input text using Groq LLM.
    
    Args:
        text: Text to detect language of
    
    Returns:
        Detected language code
    """
    if not text or not text.strip():
        return "en"

    client = get_client()

    prompt = f"""Detect the language of the following text.
Return ONLY the ISO 639-1 language code (e.g., 'en', 'es', 'fr', 'ar', 'hi').

Text: {text}

Language code:"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )
        detected = response.choices[0].message.content.strip().lower()
        # Validate it's a known code
        if detected in SUPPORTED_LANGUAGES:
            return detected
        return "en"
    except Exception:
        return "en"


def get_supported_languages() -> dict:
    """Return dictionary of supported languages."""
    return SUPPORTED_LANGUAGES.copy()


def multilingual_chat_response(user_message: str, context: str = "",
                               preferred_language: str = "en") -> str:
    """
    Generate a chat response in the user's preferred language.
    
    Args:
        user_message: User's message (in any language)
        context: Additional context about the stadium/event
        preferred_language: Language code for the response
    
    Returns:
        AI response in the preferred language
    """
    client = get_client()
    lang_name = SUPPORTED_LANGUAGES.get(preferred_language, "English")

    prompt = f"""You are a multilingual AI assistant for FIFA World Cup 2026 stadiums.
Respond in {lang_name}.

{f'Context: {context}' if context else ''}

User message: {user_message}

Provide a helpful, concise response in {lang_name}."""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"
