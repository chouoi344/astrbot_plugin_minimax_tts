"""Build Tencent SILK records for QQ without changing AstrBot's send lifecycle."""

from __future__ import annotations

import asyncio
import base64
import importlib
import io
import struct
import wave
from pathlib import Path

from .compat import import_message_components

Record, _ = import_message_components()

_TENCENT_SILK_HEADER = b"\x02#!SILK_V3"
_SILK_SAMPLE_RATE = 24000
_SILK_BITRATE = 32000


class QQVoiceError(Exception):
    """The requested QQ voice could not be prepared; callers must stop sending."""


def is_qq_event(event) -> bool:
    """Use the exact OneBot platform name; do not infer QQ from user identifiers."""
    get_platform_name = getattr(event, "get_platform_name", None)
    return callable(get_platform_name) and get_platform_name() == "aiocqhttp"


class _QQVoiceRecord(Record):
    async def convert_to_base64(self) -> str:
        """Return the prepared SILK payload without AstrBot's WAV conversion."""
        if not isinstance(self.file, str) or not self.file.startswith("base64://"):
            raise QQVoiceError("QQ 语音数据无效，无法发送。")
        return self.file[len("base64://") :]


def _validate_tencent_silk(content: bytes) -> None:
    """Check Tencent framing, including complete nonempty SILK packets."""
    if not content.startswith(_TENCENT_SILK_HEADER):
        raise QQVoiceError("QQ 语音不是有效的腾讯 SILK 格式。")
    offset = len(_TENCENT_SILK_HEADER)
    packets = 0
    while offset < len(content):
        if len(content) - offset < 2:
            raise QQVoiceError("腾讯 SILK 语音数据不完整，无法发送。")
        size = struct.unpack_from("<h", content, offset)[0]
        offset += 2
        if size == -1 and offset == len(content):
            break
        if size <= 0 or offset + size > len(content):
            raise QQVoiceError("腾讯 SILK 语音数据不完整，无法发送。")
        offset += size
        packets += 1
    if not packets:
        raise QQVoiceError("腾讯 SILK 语音没有可发送的音频数据。")


def _read_wave_pcm(content: bytes) -> bytes:
    if not (content.startswith(b"RIFF") and content[8:12] == b"WAVE"):
        raise QQVoiceError("QQ 语音只支持 PCM16、44.1kHz、单声道 WAV 或腾讯 SILK。")
    try:
        with wave.open(io.BytesIO(content), "rb") as audio:
            if (
                audio.getcomptype() != "NONE"
                or audio.getsampwidth() != 2
                or audio.getnchannels() != 1
                or audio.getframerate() != 44100
                or audio.getnframes() <= 0
            ):
                raise QQVoiceError("QQ 语音需要 PCM16、44.1kHz、单声道 WAV。")
            frames = audio.getnframes()
            pcm = audio.readframes(frames)
            if len(pcm) != frames * 2:
                raise QQVoiceError("WAV 语音数据不完整，无法转换为 QQ 语音。")
            return pcm
    except (wave.Error, EOFError, OSError, ValueError) as error:
        raise QQVoiceError("WAV 语音格式无效，无法转换为 QQ 语音。") from error


def _prepare_base64(audio_path: Path) -> str:
    try:
        content = audio_path.read_bytes()
    except OSError as error:
        raise QQVoiceError("无法读取用于 QQ 发送的语音文件。") from error
    if content.startswith(_TENCENT_SILK_HEADER):
        _validate_tencent_silk(content)
        return base64.b64encode(content).decode("ascii")

    pcm = _read_wave_pcm(content)
    try:
        pysilk = importlib.import_module("pysilk")
    except (ImportError, OSError) as error:
        raise QQVoiceError(
            "QQ 语音编码依赖不可用，请安装或更新 silk-python。"
        ) from error
    try:
        audioop = importlib.import_module("audioop")
    except (ImportError, OSError) as error:
        raise QQVoiceError(
            "QQ 语音重采样依赖不可用；Python 3.13 及以上请安装 audioop-lts。"
        ) from error
    try:
        pcm, _ = audioop.ratecv(pcm, 2, 1, 44100, _SILK_SAMPLE_RATE, None)
        # pysilk reads complete 20ms PCM frames; preserve the final partial frame.
        frame_bytes = _SILK_SAMPLE_RATE * 20 // 1000 * 2
        if remainder := len(pcm) % frame_bytes:
            pcm += b"\x00" * (frame_bytes - remainder)
        # Flush SILK encoder lookahead so the last audible samples are emitted.
        pcm += b"\x00" * frame_bytes
        output = io.BytesIO()
        pysilk.encode(
            io.BytesIO(pcm),
            output,
            sample_rate=_SILK_SAMPLE_RATE,
            bit_rate=_SILK_BITRATE,
            tencent=True,
        )
        silk = output.getvalue()
        _validate_tencent_silk(silk)
        return base64.b64encode(silk).decode("ascii")
    except QQVoiceError:
        raise
    except Exception as error:
        raise QQVoiceError("QQ 语音 SILK 编码失败，无法发送语音。") from error


async def build_voice_record(event, audio_path: str | Path) -> Record:
    """Prepare QQ audio off the event loop, leaving all sending to the caller."""
    if not is_qq_event(event):
        return Record(file=str(audio_path))
    encoded = await asyncio.to_thread(_prepare_base64, Path(audio_path))
    return _QQVoiceRecord(file=f"base64://{encoded}")
