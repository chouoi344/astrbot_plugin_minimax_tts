# -*- coding: utf-8 -*-
"""Configuration manager for the MiniMax-only TTS plugin."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
from typing import Any, Dict, List, Optional, Union

from .constants import (
    CONFIG_FILE,
    DEFAULT_API_FORMAT,
    DEFAULT_API_MAX_RETRIES,
    DEFAULT_API_SPEED,
    DEFAULT_API_TIMEOUT,
    DEFAULT_COOLDOWN,
    DEFAULT_FEATURE_MODE,
    DEFAULT_MINIMAX_BITRATE,
    DEFAULT_MINIMAX_CHANNEL,
    DEFAULT_MINIMAX_LANGUAGE_BOOST,
    DEFAULT_MINIMAX_MODEL,
    DEFAULT_MINIMAX_OUTPUT_FORMAT,
    DEFAULT_MINIMAX_PROXY,
    DEFAULT_MINIMAX_PITCH,
    DEFAULT_MINIMAX_URL,
    DEFAULT_MINIMAX_VOICE_ID,
    DEFAULT_MINIMAX_VOL,
    DEFAULT_SAMPLE_RATE_MP3_WAV,
    DEFAULT_SEGMENTED_MIN_SEGMENT_LENGTH,
    DEFAULT_SEGMENTED_OUTPUT_ENABLE,
    DEFAULT_TEXT_LIMIT,
    DEFAULT_TEXT_MIN_LIMIT,
    DEFAULT_TTS_PROVIDER,
    DEFAULT_VOICE_OUTPUT_ENABLE,
    RUNTIME_CONFIG_FILE,
    VOICE_PRESETS,
)

logger = logging.getLogger(__name__)

FEATURE_VOICE_OUTPUT = "voice_output"
FEATURE_SEGMENTED = "segmented_output"

VALID_FEATURES = {
    FEATURE_VOICE_OUTPUT,
    FEATURE_SEGMENTED,
}


def _normalize_mode(mode: Any) -> str:
    value = str(mode or DEFAULT_FEATURE_MODE).strip().lower()
    return "whitelist" if value == "whitelist" else "blacklist"


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return json.loads(json.dumps(value, ensure_ascii=False))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            logger.warning("Invalid JSON object config, fallback to empty dict")
    return {}


def _parse_json_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return json.loads(json.dumps(value, ensure_ascii=False))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            logger.warning("Invalid JSON list config, fallback to empty list")
    return []


class ConfigManager:
    """Supports AstrBotConfig and local JSON fallback."""

    VISIBLE_KEYS = ("api_key", "voice_id", "volume", "language")
    RUNTIME_KEYS = ("feature_policies", "text_limit", "text_min_limit", "cooldown", "segmented_tts")

    def __init__(self, config: Optional[Any] = None):
        self._is_astrbot_config = False
        self._config: Union[Any, Dict[str, Any]] = {}
        self._save_lock = asyncio.Lock()

        try:
            from astrbot.core.config.astrbot_config import AstrBotConfig
            if isinstance(config, AstrBotConfig):
                self._is_astrbot_config = True
                self._config = config
            else:
                self._config = config or {}
        except ImportError:
            self._config = config or {}

        # Keep internal command policies out of the four-field WebUI object.
        self._settings = self._config
        self._config = copy.deepcopy(dict(self._settings))
        if RUNTIME_CONFIG_FILE.exists():
            try:
                runtime = json.loads(RUNTIME_CONFIG_FILE.read_text(encoding="utf-8"))
                if isinstance(runtime, dict):
                    for key in self.RUNTIME_KEYS:
                        if key in runtime and key not in self._config:
                            self._config[key] = runtime[key]
            except (OSError, ValueError):
                logger.warning("Unable to load internal voice policies; using defaults")
        self._ensure_defaults()
        for key in self.VISIBLE_KEYS:
            self._settings.setdefault(key, self._config[key])
        for key in list(self._settings):
            if key not in self.VISIBLE_KEYS:
                self._settings.pop(key, None)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    async def save_async(self) -> None:
        async with self._save_lock:
            runtime = {key: self._config[key] for key in self.RUNTIME_KEYS if key in self._config}
            await asyncio.to_thread(
                RUNTIME_CONFIG_FILE.write_text,
                json.dumps(runtime, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if self._is_astrbot_config:
                if hasattr(self._settings, "save_config"):
                    await asyncio.to_thread(self._settings.save_config)
                return
            try:
                def _write():
                    CONFIG_FILE.write_text(
                        json.dumps(self._settings, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                await asyncio.to_thread(_write)
            except Exception as e:
                logger.error("Save config failed: %s", e)

    # ------------------------------------------------------------------
    # Basic dict-like APIs
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        try:
            if key in self.VISIBLE_KEYS:
                return self._settings.get(key, default)
            return self._config.get(key, default)
        except Exception:
            return default

    async def set_and_save(self, key: str, value: Any) -> None:
        if key in self.VISIBLE_KEYS:
            self._settings[key] = value
        else:
            self._config[key] = value
        await self.save_async()

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        try:
            return key in self._config
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_defaults(enable: bool) -> Dict[str, Any]:
        return {
            "enable": enable,
            "mode": DEFAULT_FEATURE_MODE,
            "enabled_umos": [],
            "disabled_umos": [],
        }

    def _ensure_defaults(self) -> None:
        # UMO guide
        raw_umo_guide = str(self._config.get("umo_guide", "") or "").strip()
        if not raw_umo_guide or "/sid" not in raw_umo_guide:
            self._config["umo_guide"] = "在聊天中发送 /sid 获取当前会话 UMO。"

        # Feature policies
        fp = self.get("feature_policies", {}) or {}
        for feat, default_enable in [
            (FEATURE_VOICE_OUTPUT, DEFAULT_VOICE_OUTPUT_ENABLE),
            (FEATURE_SEGMENTED, DEFAULT_SEGMENTED_OUTPUT_ENABLE),
        ]:
            if feat not in fp:
                fp[feat] = self._feature_defaults(default_enable)
        self._config["feature_policies"] = fp

        # Scalar defaults
        defaults = {
            "text_limit": DEFAULT_TEXT_LIMIT,
            "text_min_limit": DEFAULT_TEXT_MIN_LIMIT,
            "cooldown": DEFAULT_COOLDOWN,
        }
        for k, v in defaults.items():
            if k not in self._config:
                self._config[k] = v

        legacy_engine = self.get("tts_engine", {}) or {}
        legacy_mm = legacy_engine.get("minimax", {}) or {}
        simple_defaults = {
            "api_key": legacy_mm.get("key", ""),
            "voice_id": DEFAULT_MINIMAX_VOICE_ID,
            "volume": legacy_mm.get("vol", DEFAULT_MINIMAX_VOL),
            "language": legacy_mm.get("language_boost", DEFAULT_MINIMAX_LANGUAGE_BOOST),
        }
        for key, value in simple_defaults.items():
            if key not in self._config:
                self._config[key] = value

        # Internal defaults only; never add these fields to the WebUI object.
        engine = self.get("tts_engine", {}) or {}
        engine["provider"] = DEFAULT_TTS_PROVIDER
        if "timeout" not in engine:
            engine["timeout"] = DEFAULT_API_TIMEOUT
        if "max_retries" not in engine:
            engine["max_retries"] = DEFAULT_API_MAX_RETRIES

        mm = engine.get("minimax", {}) or {}
        mm_defaults = {
            "url": DEFAULT_MINIMAX_URL,
            "key": "",
            "model": DEFAULT_MINIMAX_MODEL,
            "voice_id": DEFAULT_MINIMAX_VOICE_ID,
            "speed": DEFAULT_API_SPEED,
            "vol": DEFAULT_MINIMAX_VOL,
            "pitch": DEFAULT_MINIMAX_PITCH,
            "audio_format": DEFAULT_API_FORMAT,
            "sample_rate": DEFAULT_SAMPLE_RATE_MP3_WAV,
            "bitrate": DEFAULT_MINIMAX_BITRATE,
            "channel": DEFAULT_MINIMAX_CHANNEL,
            "output_format": DEFAULT_MINIMAX_OUTPUT_FORMAT,
            "language_boost": DEFAULT_MINIMAX_LANGUAGE_BOOST,
            "proxy": DEFAULT_MINIMAX_PROXY,
            "voice_modify": {},
            "timbre_weights": [],
            "subtitle_enable": False,
            "pronunciation_dict": {},
            "aigc_watermark": False,
        }
        for k, v in mm_defaults.items():
            if k not in mm:
                mm[k] = v
        # 该插件的高级音频项不在 UI 展示；升级时统一迁移到高质量 WAV 源，
        # 避免 MiniMax MP3 再被 QQ 语音二次有损压缩。
        mm.pop("emotion", None)
        mm["audio_format"] = DEFAULT_API_FORMAT
        mm["sample_rate"] = DEFAULT_SAMPLE_RATE_MP3_WAV
        mm["bitrate"] = DEFAULT_MINIMAX_BITRATE
        engine["minimax"] = mm
        self._config["tts_engine"] = engine

        # 情绪路由已移除，旧配置不再参与运行。
        self._config.pop("emotion_route", None)

        # Segmented TTS
        seg = self.get("segmented_tts", {}) or {}
        seg_defaults = {
            "enable": True,
            "interval_mode": "fixed",
            "fixed_interval": 1.5,
            "adaptive_buffer": 0.5,
            "max_segments": 10,
            "min_segment_chars": 3,
            "split_pattern": r"[。？！!?\n…]+",
            "min_segment_length": DEFAULT_SEGMENTED_MIN_SEGMENT_LENGTH,
        }
        for k, v in seg_defaults.items():
            if k not in seg:
                seg[k] = v
        # Fix corrupted split_pattern (encoding issues)
        raw_sp = str(seg.get("split_pattern", ""))
        if any(c in raw_sp for c in ("銆", "鈥", "閵")):
            seg["split_pattern"] = r"[。？！!?\n…]+"
        self._config["segmented_tts"] = seg

    # ------------------------------------------------------------------
    # Feature policy APIs
    # ------------------------------------------------------------------

    def get_feature_policy(self, feature: str) -> Dict[str, Any]:
        if feature not in VALID_FEATURES:
            return self._feature_defaults(False)
        policies = self.get("feature_policies", {}) or {}
        policy = copy.deepcopy(policies.get(feature, {}))
        defaults = self._feature_defaults(False)
        defaults.update(policy)
        defaults["mode"] = _normalize_mode(defaults.get("mode"))
        defaults["enabled_umos"] = list(defaults.get("enabled_umos", []) or [])
        defaults["disabled_umos"] = list(defaults.get("disabled_umos", []) or [])
        defaults["enable"] = bool(defaults.get("enable", False))
        return defaults

    async def set_feature_policy_async(self, feature: str, policy: Dict[str, Any]) -> None:
        if feature not in VALID_FEATURES:
            return
        merged = self.get_feature_policy(feature)
        merged.update(policy or {})
        merged["mode"] = _normalize_mode(merged.get("mode"))
        merged["enabled_umos"] = list(merged.get("enabled_umos", []) or [])
        merged["disabled_umos"] = list(merged.get("disabled_umos", []) or [])
        merged["enable"] = bool(merged.get("enable", False))

        policies = self.get("feature_policies", {}) or {}
        policies[feature] = merged
        await self.set_and_save("feature_policies", policies)

    def _is_feature_enabled_for_umo(self, feature: str, umo: str) -> bool:
        policy = self.get_feature_policy(feature)
        if not policy["enable"]:
            return False
        if _normalize_mode(policy["mode"]) == "whitelist":
            return umo in policy["enabled_umos"]
        return umo not in policy["disabled_umos"]

    def is_voice_output_enabled_for_umo(self, umo: str) -> bool:
        # Tool-only distribution: persisted automatic-output settings are ignored.
        return False

    def is_segmented_output_enabled_for_umo(self, umo: str) -> bool:
        return self._is_feature_enabled_for_umo(FEATURE_SEGMENTED, umo)

    # ------------------------------------------------------------------
    # UMO list mutation (unified for all features)
    # ------------------------------------------------------------------

    async def add_umo_to_feature(self, feature: str, umo: str, list_name: str = "enabled_umos") -> None:
        policy = self.get_feature_policy(feature)
        if umo not in policy[list_name]:
            policy[list_name].append(umo)
            await self.set_feature_policy_async(feature, policy)

    async def remove_umo_from_feature(self, feature: str, umo: str, list_name: str = "enabled_umos") -> None:
        policy = self.get_feature_policy(feature)
        if umo in policy[list_name]:
            policy[list_name].remove(umo)
            await self.set_feature_policy_async(feature, policy)

    # Convenience shortcuts for voice_output
    async def add_to_enabled_umos_async(self, umo: str) -> None:
        await self.add_umo_to_feature(FEATURE_VOICE_OUTPUT, umo, "enabled_umos")

    async def remove_from_enabled_umos_async(self, umo: str) -> None:
        await self.remove_umo_from_feature(FEATURE_VOICE_OUTPUT, umo, "enabled_umos")

    async def add_to_disabled_umos_async(self, umo: str) -> None:
        await self.add_umo_to_feature(FEATURE_VOICE_OUTPUT, umo, "disabled_umos")

    async def remove_from_disabled_umos_async(self, umo: str) -> None:
        await self.remove_umo_from_feature(FEATURE_VOICE_OUTPUT, umo, "disabled_umos")

    # ------------------------------------------------------------------
    # TTS engine APIs
    # ------------------------------------------------------------------

    def get_tts_provider(self) -> str:
        return DEFAULT_TTS_PROVIDER

    def is_emotion_route_enabled(self) -> bool:
        return False

    def get_default_voice(self) -> str:
        api_cfg = self.get_api_config()
        return str(api_cfg.get("default_voice", "") or "")

    def get_api_config(self) -> Dict[str, Any]:
        voice_id = str(self.get("voice_id", DEFAULT_MINIMAX_VOICE_ID) or DEFAULT_MINIMAX_VOICE_ID).strip()
        if voice_id not in VOICE_PRESETS:
            raise ValueError("请在插件设置中重新选择音色")
        volume = _safe_float(self.get("volume", DEFAULT_MINIMAX_VOL), DEFAULT_MINIMAX_VOL)
        if not math.isfinite(volume) or not 0 < volume <= 10:
            raise ValueError("音量倍数必须大于 0 且不超过 10")
        return {
            "provider": "minimax",
            "url": DEFAULT_MINIMAX_URL,
            "key": str(self.get("api_key", "") or "").strip(),
            "model": DEFAULT_MINIMAX_MODEL,
            "voice_id": voice_id,
            "speed": DEFAULT_API_SPEED,
            "vol": volume,
            "pitch": DEFAULT_MINIMAX_PITCH,
            "format": DEFAULT_API_FORMAT,
            "sample_rate": DEFAULT_SAMPLE_RATE_MP3_WAV,
            "bitrate": DEFAULT_MINIMAX_BITRATE,
            "channel": DEFAULT_MINIMAX_CHANNEL,
            "output_format": DEFAULT_MINIMAX_OUTPUT_FORMAT,
            "language_boost": str(self.get("language", DEFAULT_MINIMAX_LANGUAGE_BOOST) or "").strip(),
            "timeout": DEFAULT_API_TIMEOUT,
            "max_retries": DEFAULT_API_MAX_RETRIES,
            "default_voice": voice_id,
        }

    # ------------------------------------------------------------------
    # Emotion route + marker
    # ------------------------------------------------------------------

    def get_voice_map(self) -> Dict[str, str]:
        return {}

    def get_speed_map(self) -> Dict[str, float]:
        return {}

    def get_marker_config(self) -> Dict[str, Any]:
        return {}

    def is_marker_enabled(self) -> bool:
        return False

    def get_marker_tag(self) -> str:
        return "EMO"

    def get_marker_prompt_hint(self) -> str:
        return ""

    def get_emotion_keywords(self) -> Dict[str, List[str]]:
        return {}

    # ------------------------------------------------------------------
    # Scalar getters
    # ------------------------------------------------------------------

    def get_global_enable(self) -> bool:
        policy = self.get_feature_policy(FEATURE_VOICE_OUTPUT)
        return bool(policy["enable"] and policy["mode"] == "blacklist")

    def get_enabled_umos(self) -> List[str]:
        return self.get_feature_policy(FEATURE_VOICE_OUTPUT)["enabled_umos"]

    def get_disabled_umos(self) -> List[str]:
        return self.get_feature_policy(FEATURE_VOICE_OUTPUT)["disabled_umos"]

    def get_text_limit(self) -> int:
        return _safe_int(self.get("text_limit", DEFAULT_TEXT_LIMIT), DEFAULT_TEXT_LIMIT)

    def get_text_min_limit(self) -> int:
        return _safe_int(self.get("text_min_limit", DEFAULT_TEXT_MIN_LIMIT), DEFAULT_TEXT_MIN_LIMIT)

    def get_cooldown(self) -> int:
        return _safe_int(self.get("cooldown", DEFAULT_COOLDOWN), DEFAULT_COOLDOWN)

    # ------------------------------------------------------------------
    # Async setters (no sync duplicates)
    # ------------------------------------------------------------------

    async def set_voice_output_enable_async(self, enable: bool) -> None:
        await self.set_feature_policy_async(FEATURE_VOICE_OUTPUT, {"enable": bool(enable)})

    async def set_marker_enable_async(self, enable: bool) -> None:
        route = self.get("emotion_route", {}) or {}
        marker = route.get("marker", {}) or {}
        marker["enable"] = bool(enable)
        route["marker"] = marker
        await self.set_and_save("emotion_route", route)

    # ------------------------------------------------------------------
    # Segmented TTS
    # ------------------------------------------------------------------

    def get_segmented_tts_config(self) -> Dict[str, Any]:
        return self.get("segmented_tts", {}) or {}

    def is_segmented_tts_enabled(self) -> bool:
        return bool(self.get_segmented_tts_config().get("enable", False))

    def get_segmented_tts_interval_mode(self) -> str:
        mode = str(self.get_segmented_tts_config().get("interval_mode", "fixed"))
        return mode if mode in ("fixed", "adaptive") else "fixed"

    def get_segmented_tts_fixed_interval(self) -> float:
        return _safe_float(self.get_segmented_tts_config().get("fixed_interval"), 1.5)

    def get_segmented_tts_adaptive_buffer(self) -> float:
        return _safe_float(self.get_segmented_tts_config().get("adaptive_buffer"), 0.5)

    def get_segmented_tts_max_segments(self) -> int:
        return _safe_int(self.get_segmented_tts_config().get("max_segments"), 10)

    def get_segmented_tts_min_segment_chars(self) -> int:
        return _safe_int(self.get_segmented_tts_config().get("min_segment_chars"), 3)

    def get_segmented_tts_split_pattern(self) -> str:
        return str(self.get_segmented_tts_config().get("split_pattern", r"[。？！!?\n…]+"))

    def get_segmented_tts_min_segment_length(self) -> int:
        return _safe_int(
            self.get_segmented_tts_config().get("min_segment_length"),
            DEFAULT_SEGMENTED_MIN_SEGMENT_LENGTH,
        )
