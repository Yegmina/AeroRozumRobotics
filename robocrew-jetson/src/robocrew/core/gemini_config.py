from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


DEFAULT_GEMINI_ROBOTICS_MODEL = "gemini-robotics-er-1.6-preview"
DEFAULT_GEMINI_REASONING_MODEL = "gemini-3.1-pro-preview"
DEFAULT_GEMINI_SPEAKING_MODEL = "gemini-3.5-flash"
DEFAULT_GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_GEMINI_TTS_VOICE = "Zephyr"

_ENV_LOADED = False


def load_gemini_env() -> None:
    """Load Gemini env vars from normal locations plus the sibling robotics repo."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    repo_root = Path(__file__).resolve().parents[3]
    env_paths = [
        Path.cwd() / ".env",
        Path(find_dotenv(usecwd=True)) if find_dotenv(usecwd=True) else None,
        repo_root / ".env",
        repo_root.parent / ".env",
        repo_root.parent / "AeroRozumRobotics" / ".env",
        repo_root.parent / "AeroRozumRobotics" / "gemini-robot-command-agent" / ".env",
    ]

    seen: set[Path] = set()
    for env_path in env_paths:
        if env_path is None:
            continue
        resolved = env_path.expanduser().resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        load_dotenv(resolved, override=False)

    gemini_key = os.environ.get("GEMINI_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")
    if gemini_key and not google_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key
    elif google_key and not gemini_key:
        os.environ["GEMINI_API_KEY"] = google_key

    _ENV_LOADED = True


def get_gemini_api_key(required: bool = False) -> str | None:
    load_gemini_env()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if required and not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in environment or .env")
    return api_key


def get_gemini_model(env_var: str, default_model: str) -> str:
    load_gemini_env()
    return os.environ.get(env_var, default_model)


def get_langchain_gemini_model(env_var: str, default_model: str) -> str:
    model = get_gemini_model(env_var, default_model)
    if ":" in model:
        return model
    return f"google_genai:{model}"


def get_gemini_robotics_langchain_model() -> str:
    return get_langchain_gemini_model("GEMINI_ROBOTICS_MODEL", DEFAULT_GEMINI_ROBOTICS_MODEL)


def get_gemini_reasoning_langchain_model() -> str:
    return get_langchain_gemini_model("GEMINI_REASONING_MODEL", DEFAULT_GEMINI_REASONING_MODEL)


def get_gemini_speaking_langchain_model() -> str:
    return get_langchain_gemini_model("GEMINI_SPEAKING_MODEL", DEFAULT_GEMINI_SPEAKING_MODEL)


def get_gemini_tts_model() -> str:
    return get_gemini_model("GEMINI_TTS_MODEL", DEFAULT_GEMINI_TTS_MODEL)


def get_gemini_tts_voice() -> str:
    return get_gemini_model("GEMINI_TTS_VOICE", DEFAULT_GEMINI_TTS_VOICE)
