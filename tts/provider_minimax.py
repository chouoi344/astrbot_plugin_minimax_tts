import asyncio
import base64
import copy
import hashlib
import json
import logging
import tempfile
import weakref
from pathlib import Path
from typing import Any, Optional

import aiohttp

from ..utils.audio import validate_audio_file


logger = logging.getLogger(__name__)


class MiniMaxTTS:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        *,
        fmt: str = "wav",
        speed: float = 1.0,
        voice_id: str = "",
        vol: float = 1.0,
        pitch: int = 0,
        sample_rate: int = 44100,
        bitrate: int = 256000,
        channel: int = 1,
        output_format: str = "hex",
        language_boost: str = "",
        proxy: str = "",
        voice_modify: Optional[dict] = None,
        timber_weights: Optional[list] = None,
        subtitle_enable: bool = False,
        pronunciation_dict: Optional[dict] = None,
        aigc_watermark: bool = False,
        max_retries: int = 2,
        timeout: int = 30,
    ):
        self.api_url = api_url.strip() or "https://choupi.tech/v1/audio/speech"
        self.api_key = api_key.strip()
        self.model = model or "speech-2.8-hd"
        self.format = (fmt or "mp3").lower()
        self.speed = float(speed)
        self.voice_id = voice_id or ""
        self.vol = float(vol)
        self.pitch = int(pitch)
        self.sample_rate = int(sample_rate)
        self.bitrate = int(bitrate)
        self.channel = int(channel)
        self.output_format = str(output_format or "hex").strip().lower() or "hex"
        self.language_boost = str(language_boost or "").strip()
        self.proxy = str(proxy or "").strip() or None
        self.voice_modify = copy.deepcopy(voice_modify or {})
        self.timber_weights = copy.deepcopy(timber_weights or [])
        self.subtitle_enable = bool(subtitle_enable)
        self.pronunciation_dict = copy.deepcopy(pronunciation_dict or {})
        self.aigc_watermark = bool(aigc_watermark)
        self.max_retries = max(0, int(max_retries))
        self.timeout = max(5, int(timeout))
        self.transport_mode = "sync_http"
        self.last_response_meta: Optional[dict[str, Any]] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            client_timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=client_timeout)

    @staticmethod
    def _looks_like_hex(s: str) -> bool:
        if len(s) < 4 or len(s) % 2 != 0:
            return False
        try:
            int(s[:16], 16)
            return all(c in "0123456789abcdefABCDEF" for c in s[:64])
        except Exception:
            return False

    @staticmethod
    async def _write_bytes(path: Path, content: bytes) -> None:
        def _write():
            with open(path, "wb") as f:
                f.write(content)

        await asyncio.to_thread(_write)

    async def _download_to_path(self, url: str, out_path: Path) -> bool:
        if not url:
            return False
        await self._ensure_session()
        try:
            assert self._session is not None
            async with self._session.get(url, proxy=self.proxy) as response:
                if response.status != 200:
                    return False
                content = await response.read()
                if not content:
                    return False
                await self._write_bytes(out_path, content)
                return True
        except Exception:
            return False

    def _build_sync_http_payload(
        self,
        text: str,
        *,
        voice: str,
        speed: float,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "voice_setting": {
                "speed": speed,
                "vol": self.vol,
                "pitch": self.pitch,
            },
            "audio_setting": {
                "sample_rate": self.sample_rate,
                "channel": self.channel,
            },
        }
        if self.format == "mp3":
            metadata["audio_setting"]["bitrate"] = self.bitrate
        if self.language_boost:
            metadata["language_boost"] = self.language_boost
        return {
            "model": self.model,
            "input": text,
            "voice": voice,
            "speed": speed,
            "response_format": self.format,
            "metadata": metadata,
        }

    def _extract_response_meta(self, data: dict[str, Any]) -> dict[str, Any]:
        body = data.get("data", {}) or {}
        extra_info = body.get("extra_info")
        if not isinstance(extra_info, dict):
            extra_info = data.get("extra_info") if isinstance(data.get("extra_info"), dict) else {}

        return {
            "status_code": (data.get("base_resp") or {}).get("status_code"),
            "status_msg": (data.get("base_resp") or {}).get("status_msg"),
            "usage_characters": extra_info.get("usage_characters"),
            "audio_length": extra_info.get("audio_length"),
            "invisible_character_ratio": extra_info.get("invisible_character_ratio"),
        }

    def _log_response_meta(self, meta: dict[str, Any]) -> None:
        compact_meta = {key: value for key, value in meta.items() if value not in (None, "", [], {})}
        if compact_meta:
            logger.info("MiniMaxTTS response meta: %s", compact_meta)

    async def synth(
        self,
        text: str,
        voice: str,
        out_dir: Path,
        speed: Optional[float] = None,
    ) -> Optional[Path]:
        if not self.api_key:
            logger.error("MiniMaxTTS: missing api key")
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = self._build_sync_http_payload(
            text, voice=voice or self.voice_id,
            speed=float(speed) if speed is not None else self.speed,
        )
        cache_key = hashlib.sha256(json.dumps(
            {"url": self.api_url, "credential": hashlib.sha256(self.api_key.encode()).hexdigest(),
             "payload": payload}, ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")).hexdigest()[:32]
        out_path = out_dir / f"{cache_key}.{self.format}"
        lock = self._cache_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            if out_path.exists():
                if await validate_audio_file(out_path, expected_format=self.format):
                    return out_path
                out_path.unlink(missing_ok=True)
            return await self._synth_uncached(payload, out_path)

    async def _synth_uncached(self, payload: dict[str, Any], out_path: Path) -> Optional[Path]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        await self._ensure_session()
        temp_path = None
        last_error = "no response"
        for attempt in range(self.max_retries + 1):
            try:
                assert self._session is not None
                async with self._session.post(
                    self.api_url, headers=headers, json=payload, proxy=self.proxy,
                    allow_redirects=False,
                ) as resp:
                    self.last_response_meta = {"http_status": resp.status}
                    if resp.status == 429 and attempt < self.max_retries:
                        await asyncio.sleep(min(2 ** attempt, 8))
                        continue
                    if not 200 <= resp.status < 300:
                        last_error = f"http {resp.status}"
                        break
                    # Inspect the actual WAV, not the relay's Content-Type.
                    raw = await resp.read()
                    if not raw:
                        last_error = "empty audio response"
                        break
                    with tempfile.NamedTemporaryFile(
                        dir=out_path.parent, prefix="tts_", suffix=f".{self.format}", delete=False,
                    ) as temp_file:
                        temp_path = Path(temp_file.name)
                    await self._write_bytes(temp_path, raw)
                    if not await validate_audio_file(temp_path, expected_format=self.format):
                        last_error = "relay did not return valid PCM WAV audio"
                        break
                    await asyncio.to_thread(temp_path.replace, out_path)
                    temp_path = None
                    self.last_response_meta["audio_bytes"] = len(raw)
                    return out_path
            except aiohttp.ClientConnectorError:
                # No connection was established, so retrying cannot repeat a
                # completed synthesis. Never retry read timeouts/5xx/parsing.
                last_error = "connection failed"
                if attempt < self.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue
                break
            except Exception as exc:
                last_error = type(exc).__name__
                break
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
                    temp_path = None
        logger.error("MiniMaxTTS synth failed: %s", last_error)
        return None
