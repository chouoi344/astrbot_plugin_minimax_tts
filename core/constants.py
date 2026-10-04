# -*- coding: utf-8 -*-
"""Shared constants for the MiniMax TTS plugin."""

from pathlib import Path
from typing import Dict, List, Set, Tuple


# Plugin metadata
PLUGIN_ID = "astrbot_plugin_minimax_tts"
PLUGIN_NAME = "MiniMax 语音助手"
PLUGIN_DESC = "通过终逢小站调用 MiniMax，内置角色音色，支持高质量语音、分段发送和语言选择。"
PLUGIN_VERSION = "0.1.9"
PLUGIN_AUTHOR = "臭屁"

# Paths
PLUGIN_DIR = Path(__file__).parent.parent
CONFIG_FILE = PLUGIN_DIR / "config.json"
RUNTIME_CONFIG_FILE = PLUGIN_DIR / "runtime_config.json"
TEMP_DIR = PLUGIN_DIR / "temp"

# Emotion constants
EMOTIONS: Tuple[str, ...] = ("happy", "sad", "angry", "neutral")

INVISIBLE_CHARS: List[str] = [
    "\ufeff",
    "\u200b",
    "\u200c",
    "\u200d",
    "\u200e",
    "\u200f",
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
]

EMOTION_SYNONYMS: Dict[str, Set[str]] = {
    "happy": {"happy", "joy", "joyful", "cheerful", "excited", "positive"},
    "sad": {"sad", "sorrow", "depressed", "down", "unhappy", "upset"},
    "angry": {"angry", "mad", "furious", "annoyed", "irritated", "rage"},
    "neutral": {"neutral", "calm", "normal", "objective", "ok", "fine", "confused"},
}

EMOTION_PREFERENCE_MAP: Dict[str, str] = {
    "sad": "neutral",
    "angry": "neutral",
    "happy": "happy",
    "neutral": "neutral",
}

# Audio constants
AUDIO_CLEANUP_TTL_SECONDS: int = 2 * 3600
AUDIO_MIN_VALID_SIZE: int = 100
AUDIO_VALID_EXTENSIONS: List[str] = [".mp3", ".wav", ".opus", ".pcm", ".flac"]

# Runtime cleanup limits
SESSION_CLEANUP_INTERVAL_SECONDS: int = 1800
SESSION_MAX_IDLE_SECONDS: int = 86400
SESSION_MAX_COUNT: int = 3000
INFLIGHT_SIG_TTL_SECONDS: int = 180
INFLIGHT_SIG_MAX_COUNT: int = 2000

# MiniMax defaults
DEFAULT_TTS_PROVIDER: str = "minimax"
DEFAULT_MINIMAX_URL: str = "https://choupi.tech/v1/audio/speech"
DEFAULT_MINIMAX_MODEL: str = "speech-2.8-hd"
DEFAULT_MINIMAX_VOICE_ID: str = "minizhizhi"
VOICE_PRESETS: Dict[str, str] = {"minizhizhi": "小乌鸦"}
DEFAULT_MINIMAX_VOL: float = 1.0
DEFAULT_MINIMAX_PITCH: int = 0
DEFAULT_MINIMAX_BITRATE: int = 256000
DEFAULT_MINIMAX_CHANNEL: int = 1
DEFAULT_MINIMAX_OUTPUT_FORMAT: str = "hex"
DEFAULT_MINIMAX_LANGUAGE_BOOST: str = ""
DEFAULT_MINIMAX_PROXY: str = ""
DEFAULT_API_FORMAT: str = "wav"
DEFAULT_API_SPEED: float = 1.0
DEFAULT_API_TIMEOUT: int = 30
DEFAULT_API_MAX_RETRIES: int = 2
DEFAULT_SAMPLE_RATE_MP3_WAV: int = 44100

MINIMAX_EXPRESSIVE_MODELS: Tuple[str, ...] = ("speech-2.8-hd", "speech-2.8-turbo")
MINIMAX_EXPRESSIVE_TAGS: Tuple[str, ...] = (
    "laughs",
    "chuckle",
    "coughs",
    "clear-throat",
    "groans",
    "breath",
    "pant",
    "inhale",
    "exhale",
    "gasps",
    "sniffs",
    "sighs",
    "snorts",
    "burps",
    "lip-smacking",
    "humming",
    "hissing",
    "emm",
    "sneezes",
)

# Feature defaults
DEFAULT_FEATURE_MODE: str = "blacklist"
DEFAULT_VOICE_OUTPUT_ENABLE: bool = False
DEFAULT_SEGMENTED_OUTPUT_ENABLE: bool = True

# Runtime defaults
DEFAULT_TEXT_LIMIT: int = 50
DEFAULT_TEXT_MIN_LIMIT: int = 3
DEFAULT_COOLDOWN: int = 0
DEFAULT_EMO_MARKER_TAG: str = "EMO"
DEFAULT_SEGMENTED_MIN_SEGMENT_LENGTH: int = 3

DEFAULT_EMOTION_KEYWORDS_LIST: Dict[str, List[str]] = {
    "happy": ["开心", "高兴", "喜悦", "棒", "太好了", "great", "awesome", "nice"],
    "sad": ["难过", "伤心", "抱歉", "遗憾", "sad", "sorry", "upset", "cry"],
    "angry": ["生气", "愤怒", "恼火", "气死", "angry", "mad", "annoyed", "rage"],
}

# Misc
DEFAULT_TEST_TEXT: str = "这是一条 MiniMax TTS 测试语音。"
