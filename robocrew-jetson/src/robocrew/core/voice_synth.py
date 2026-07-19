from __future__ import annotations

import os
import sys
import subprocess
import urllib.request
import time
import struct
import wave
from pathlib import Path

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from robocrew.core.gemini_config import (
    get_gemini_api_key,
    get_gemini_tts_model,
    get_gemini_tts_voice,
    load_gemini_env,
)

DATA_DIR = Path(os.environ.get("ROBOCREW_DATA_DIR", Path.home() / ".cache" / "robocrew")).expanduser()
MODEL_PATH = DATA_DIR / "en_amy.onnx"
CONFIG_PATH = DATA_DIR / "en_amy.onnx.json"
OUTPUT_WAV = DATA_DIR / "speech.wav"


def setup_voice():
    if not DATA_DIR.exists():
        print(f"Creating folder: {DATA_DIR}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Download model if not exists or file is corrupted (too small)
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 100:
        print(f"Downloading model to {MODEL_PATH}...")
        model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low/en_US-amy-low.onnx"
        urllib.request.urlretrieve(model_url, str(MODEL_PATH))
    # Download config
    if not CONFIG_PATH.exists() or CONFIG_PATH.stat().st_size < 100:
        print(f"Downloading config to {CONFIG_PATH}...")
        config_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low/en_US-amy-low.onnx.json"
        urllib.request.urlretrieve(config_url, str(CONFIG_PATH))


def speak_and_play(text):
    """Synthesize text to speech and play it immediately."""
    audio_path = synthesize_speech(text)
    play_audio(audio_path)


def synthesize_speech(text: str) -> Path:
    load_gemini_env()
    provider = os.environ.get("ROBOCREW_TTS_PROVIDER")
    if provider is None:
        provider = "gemini" if get_gemini_api_key(required=False) else "piper"

    if provider.lower() == "gemini":
        return synthesize_gemini_speech(text)
    if provider.lower() == "piper":
        return synthesize_piper_speech(text)
    raise ValueError("ROBOCREW_TTS_PROVIDER must be 'gemini' or 'piper'")


def synthesize_piper_speech(text: str) -> Path:
    setup_voice()
    subprocess.run([
        sys.executable, "-m", "piper",
        "--model", str(MODEL_PATH),
        "--config", str(CONFIG_PATH),
        "--output_file", str(OUTPUT_WAV)
    ], input=text.encode('utf-8'), check=True, capture_output=True)
    return OUTPUT_WAV


def synthesize_gemini_speech(text: str) -> Path:
    from google import genai
    from google.genai import types

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=get_gemini_api_key(required=True))
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=f"Read aloud exactly this text:\n{text}")],
        )
    ]
    config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=get_gemini_tts_voice(),
                )
            )
        ),
    )

    audio_chunks: list[bytes] = []
    mime_type = ""
    for chunk in client.models.generate_content_stream(
        model=get_gemini_tts_model(),
        contents=contents,
        config=config,
    ):
        for part in _iter_response_parts(chunk):
            inline_data = getattr(part, "inline_data", None)
            if inline_data and inline_data.data:
                audio_chunks.append(inline_data.data)
                mime_type = inline_data.mime_type or mime_type

    if not audio_chunks:
        raise RuntimeError("Gemini TTS returned no audio")

    audio_data = b"".join(audio_chunks)
    if not _is_wav_mime_type(mime_type):
        audio_data = convert_to_wav(audio_data, mime_type)

    OUTPUT_WAV.write_bytes(audio_data)
    return OUTPUT_WAV


def play_audio(audio_path: Path) -> None:
    import pygame

    pygame.mixer.init(frequency=_wav_sample_rate(audio_path) or 24000)
    pygame.mixer.music.load(str(audio_path))
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    pygame.mixer.quit()


def _is_wav_mime_type(mime_type: str) -> bool:
    mime_type = (mime_type or "").split(";", 1)[0].strip().lower()
    return mime_type in {"audio/wav", "audio/wave", "audio/x-wav"}


def _iter_response_parts(response):
    direct_parts = getattr(response, "parts", None)
    if direct_parts:
        yield from direct_parts
        return

    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            yield part


def _wav_sample_rate(audio_path: Path) -> int | None:
    try:
        with wave.open(str(audio_path), "rb") as wav:
            return wav.getframerate()
    except wave.Error:
        return None


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Generate a WAV header for raw PCM audio returned by Gemini TTS."""
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + audio_data


def parse_audio_mime_type(mime_type: str) -> dict[str, int]:
    bits_per_sample = 16
    rate = 24000

    for part in (mime_type or "").split(";"):
        part = part.strip()
        if part.lower().startswith("rate="):
            try:
                rate = int(part.split("=", 1)[1])
            except (ValueError, IndexError):
                pass
        elif part.lower().startswith("audio/l"):
            try:
                bits_per_sample = int(part.lower().split("audio/l", 1)[1])
            except (ValueError, IndexError):
                pass

    return {"bits_per_sample": bits_per_sample, "rate": rate}

