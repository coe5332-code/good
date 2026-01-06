"""
Audio utilities for BSK training video generation

Goals:
- Clear, slow, professional narration
- Natural pauses between bullet points
- Predictable duration for video sync
"""

import tempfile
import asyncio
import re
import os

# Try to import the preferred `edge_tts` package. If it's not available,
# fall back to `gtts` if present, otherwise raise a helpful error when
# text-to-speech is requested. This makes the Streamlit UI show a clear
# message instead of crashing on import.
try:
    import edge_tts  # type: ignore
    HAS_EDGE_TTS = True
except Exception:
    edge_tts = None
    HAS_EDGE_TTS = False

try:
    from gtts import gTTS  # type: ignore
    HAS_GTTS = True
except Exception:
    gTTS = None
    HAS_GTTS = False

# -------------------------------------------------
# DEFAULT VOICE SETTINGS (TRAINING OPTIMIZED)
# -------------------------------------------------
DEFAULT_VOICE = "en-IN-NeerjaNeural"  # Indian English, calm & professional
DEFAULT_RATE = "+5%"  # Slightly slower than normal
DEFAULT_PITCH = "+0Hz"


# -------------------------------------------------
# TEXT PRE-PROCESSING (VERY IMPORTANT)
# -------------------------------------------------
def prepare_narration_text(text: str) -> str:
    """
    Convert bullet-style text into narration-friendly speech.
    Adds pauses and removes visual-only symbols.
    """

    # Remove bullet symbols if present
    text = re.sub(r"[•▪◦]", "", text)

    # Ensure proper spacing after sentences
    text = re.sub(r"\.\s*", ". ", text)

    # Add slight pause markers after each sentence
    # Edge TTS respects commas and sentence breaks well
    text = text.replace(".", ". ")

    return text.strip()


# -------------------------------------------------
# TEXT TO SPEECH (ASYNC)
# -------------------------------------------------
async def text_to_speech(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: str = DEFAULT_RATE,
    pitch: str = DEFAULT_PITCH,
):
    """
    Generate narration audio for one slide.

    Input:
    - text: narration text (usually slide bullets joined)
    Output:
    - path to generated .mp3 file
    """

    narration_text = prepare_narration_text(text)

    # If edge_tts is available, use its async API.
    if HAS_EDGE_TTS:
        communicate = edge_tts.Communicate(
            text=narration_text, voice=voice, rate=rate, pitch=pitch
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
            output_path = audio_file.name

        await communicate.save(output_path)

    # Otherwise, if gTTS is available, use it as a synchronous fallback
    # wrapped in a thread to avoid blocking the event loop.
    elif HAS_GTTS:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
            output_path = audio_file.name

        def _write_gtts():
            tts = gTTS(text=narration_text, lang="en")
            tts.save(output_path)

        await asyncio.to_thread(_write_gtts)

    else:
        # Helpful error to surface in Streamlit rather than import-time crash.
        raise ModuleNotFoundError(
            "Neither `edge_tts` nor `gtts` (gTTS) is installed.\n"
            "Install with `pip install edge-tts` (preferred) or `pip install gTTS` "
            "to enable text-to-speech."
        )

    # -------- HARD VALIDATION --------
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
        raise RuntimeError("TTS failed: empty or invalid audio file generated")

    return output_path



# -------------------------------------------------
# SYNC HELPER (OPTIONAL BUT USEFUL)
# -------------------------------------------------
def estimate_audio_duration(text: str) -> float:
    """
    Rough duration estimation (seconds),
    useful for debugging or future timing logic.
    """
    words = len(text.split())
    avg_wpm = 130  # training-friendly speed
    return (words / avg_wpm) * 60
