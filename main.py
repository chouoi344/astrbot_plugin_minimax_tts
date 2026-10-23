# -*- coding: utf-8 -*-
"""MiniMax-only TTS plugin entry."""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

from astrbot.api import logger

from .core.compat import initialize_compat

initialize_compat()

from .core.compat import (
    import_astr_message_event,
    import_astrbot_config,
    import_context_and_star,
    import_filter,
    import_llm_response,
    import_message_components,
    import_result_content_type,
)

AstrMessageEvent = import_astr_message_event()
AstrBotConfig = import_astrbot_config()
filter = import_filter()
Context, Star, register = import_context_and_star()
LLMResponse = import_llm_response()
ResultContentType = import_result_content_type()
Record, Plain = import_message_components()

from .core.config import ConfigManager
from .core.constants import (
    AUDIO_CLEANUP_TTL_SECONDS,
    DEFAULT_TEST_TEXT,
    INFLIGHT_SIG_MAX_COUNT,
    INFLIGHT_SIG_TTL_SECONDS,
    MINIMAX_EXPRESSIVE_MODELS,
    MINIMAX_EXPRESSIVE_TAGS,
    PLUGIN_DESC,
    PLUGIN_ID,
    PLUGIN_NAME,
    PLUGIN_VERSION,
    SESSION_CLEANUP_INTERVAL_SECONDS,
    SESSION_MAX_COUNT,
    SESSION_MAX_IDLE_SECONDS,
    TEMP_DIR,
)
from .core.marker import EmotionMarkerProcessor
from .core.segmented_tts import SegmentedTTSProcessor
from .core.session import SessionState
from .core.text_splitter import TextSplitter
from .core.tts_processor import TTSConditionChecker, TTSProcessor
from .tts.provider_minimax import MiniMaxTTS
from .utils.audio import cleanup_dir, ensure_dir
from .utils.extract import CodeAndLinkExtractor
from .utils.text_sanitizer import PreparedSpeechText, SpeechTextSanitizer

OUTPUT_MARKER_MODE_EXTRA = "_minimax_tts_output_marker_mode"
OUTPUT_MARKER_MODE_PRESERVE = "preserve_for_tts"
OUTPUT_MARKER_MODE_STRIP = "strip_visible"


@register(PLUGIN_ID, PLUGIN_NAME, PLUGIN_DESC, PLUGIN_VERSION)
class MiniMaxTTSPlugin(Star):
    def __init__(self, context: Context, config: Optional[dict] = None):
        super().__init__(context)
        self._session_state: Dict[str, SessionState] = {}
        self._inflight_sigs: Dict[str, float] = {}
        self._background_tasks: List[asyncio.Task] = []
        self._cleanup_task_started = False

        self._init_config(config)
        self._init_components()
        ensure_dir(TEMP_DIR)

    async def terminate(self):
        background_tasks = list(self._background_tasks)
        for task in background_tasks:
            if not task.done():
                task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        # Client-close tasks are finite work: finish them instead of cancelling.
        client_close_tasks = getattr(self, "_client_close_tasks", [])
        if client_close_tasks:
            await asyncio.gather(*list(client_close_tasks), return_exceptions=True)
        client_close_tasks.clear()

        if hasattr(self, "tts_client"):
            try:
                await self.tts_client.close()
            except Exception:
                logger.debug("close tts client failed", exc_info=True)

        self._session_state.clear()
        self._inflight_sigs.clear()

    def _init_config(self, config: Optional[dict]) -> None:
        if isinstance(config, AstrBotConfig):
            self.config = ConfigManager(config)
        else:
            self.config = ConfigManager(config or {})

        self.voice_map, self.speed_map = self._resolve_route_maps()
        self.text_limit = self.config.get_text_limit()
        self.text_min_limit = self.config.get_text_min_limit()
        self.cooldown = self.config.get_cooldown()
        self.segmented_tts_enabled = self.config.is_segmented_tts_enabled()
        self.segmented_min_chars = self.config.get_segmented_tts_min_segment_chars()

    def _resolve_route_maps(self) -> Tuple[Dict[str, str], Dict[str, float]]:
        api_cfg = self.config.get_api_config()
        default_voice = self.config.get_default_voice() or api_cfg.get("voice_id", "")
        voice_map = {"neutral": str(default_voice)} if default_voice else {}
        speed_map = {"neutral": float(api_cfg.get("speed", 1.0))}
        return voice_map, speed_map

    def _create_tts_client(self) -> MiniMaxTTS:
        api_cfg = self.config.get_api_config()
        return MiniMaxTTS(
            api_url=api_cfg["url"],
            api_key=api_cfg["key"],
            model=api_cfg["model"],
            fmt=api_cfg["format"],
            speed=api_cfg["speed"],
            voice_id=api_cfg.get("voice_id", ""),
            vol=api_cfg.get("vol", 1.0),
            pitch=api_cfg.get("pitch", 0),
            sample_rate=api_cfg.get("sample_rate", 44100),
            bitrate=api_cfg.get("bitrate", 256000),
            channel=api_cfg.get("channel", 1),
            output_format=api_cfg.get("output_format", "hex"),
            language_boost=api_cfg.get("language_boost", ""),
            proxy=api_cfg.get("proxy", ""),
            voice_modify=api_cfg.get("voice_modify", {}),
            timber_weights=api_cfg.get(
                "timber_weights", api_cfg.get("timbre_weights", [])
            ),
            subtitle_enable=api_cfg.get("subtitle_enable", False),
            pronunciation_dict=api_cfg.get("pronunciation_dict", {}),
            aigc_watermark=api_cfg.get("aigc_watermark", False),
            max_retries=api_cfg.get("max_retries", 2),
            timeout=api_cfg.get("timeout", 30),
        )

    def _get_tts_engine_signature(self) -> Tuple:
        api_cfg = self.config.get_api_config()
        keys = (
            "url",
            "key",
            "model",
            "format",
            "speed",
            "voice_id",
            "vol",
            "pitch",
            "sample_rate",
            "bitrate",
            "channel",
            "output_format",
            "language_boost",
            "proxy",
            "voice_modify",
            "timber_weights",
            "pronunciation_dict",
            "aigc_watermark",
            "subtitle_enable",
            "timeout",
            "max_retries",
        )
        return tuple((key, str(api_cfg.get(key))) for key in keys)

    def _init_components(self) -> None:
        self.tts_client = self._create_tts_client()

        self.emo_marker_enable = False
        self.marker_processor = EmotionMarkerProcessor(tag="EMO", enabled=False)

        self.extractor = CodeAndLinkExtractor()
        self.text_sanitizer = SpeechTextSanitizer(
            marker_processor=self.marker_processor,
            extractor=self.extractor,
        )
        self.tts_processor = TTSProcessor(
            tts_client=self.tts_client,
            voice_map=self.voice_map,
            speed_map=self.speed_map,
        )
        self.condition_checker = TTSConditionChecker(
            prob=1.0,
            text_limit=self.text_limit,
            text_min_limit=self.text_min_limit,
            cooldown=self.cooldown,
            allow_mixed=False,
        )
        self._init_segmented_tts()
        self._tts_engine_signature = self._get_tts_engine_signature()

    def _init_segmented_tts(self) -> None:
        splitter = TextSplitter(
            split_pattern=self.config.get_segmented_tts_split_pattern(),
            smart_mode=True,
            max_segments=self.config.get_segmented_tts_max_segments(),
            min_segment_length=self.config.get_segmented_tts_min_segment_length(),
        )
        self.segmented_tts_processor = SegmentedTTSProcessor(
            tts_processor=self.tts_processor,
            splitter=splitter,
            interval_mode=self.config.get_segmented_tts_interval_mode(),
            fixed_interval=self.config.get_segmented_tts_fixed_interval(),
            adaptive_buffer=self.config.get_segmented_tts_adaptive_buffer(),
            max_segments=self.config.get_segmented_tts_max_segments(),
            min_segment_length=self.config.get_segmented_tts_min_segment_length(),
        )

    def _update_components_from_config(self) -> None:
        self.text_limit = self.config.get_text_limit()
        self.text_min_limit = self.config.get_text_min_limit()
        self.cooldown = self.config.get_cooldown()
        self.segmented_tts_enabled = self.config.is_segmented_tts_enabled()
        self.segmented_min_chars = self.config.get_segmented_tts_min_segment_chars()

        new_signature = self._get_tts_engine_signature()
        if new_signature != getattr(self, "_tts_engine_signature", None):
            old_client = self.tts_client
            self.tts_client = self._create_tts_client()
            self.tts_processor.tts = self.tts_client
            self._tts_engine_signature = new_signature
            if old_client is not None and old_client is not self.tts_client:
                self._schedule_client_close(old_client)

        self.condition_checker.text_limit = self.text_limit
        self.condition_checker.text_min_limit = self.text_min_limit
        self.condition_checker.cooldown = self.cooldown

        self.voice_map, self.speed_map = self._resolve_route_maps()
        self.tts_processor.voice_map = self.voice_map
        self.tts_processor.speed_map = self.speed_map

        self.emo_marker_enable = False
        self.marker_processor.update_config("EMO", False)
        self._init_segmented_tts()

    def _sess_id(self, event: AstrMessageEvent) -> str:
        try:
            group_id = event.get_group_id()
        except Exception:
            group_id = ""

        if group_id and group_id not in ("", "None", "null", "0"):
            return f"group_{group_id}"
        return f"user_{event.get_sender_id()}"

    def _get_umo(self, event: AstrMessageEvent) -> str:
        try:
            umo = str(getattr(event, "unified_msg_origin", "") or "").strip()
            if umo:
                return umo
        except Exception:
            pass
        return self._sess_id(event)

    def _get_session_state(self, sid: str) -> SessionState:
        return self._session_state.setdefault(sid, SessionState())

    def _track_background_task(
        self, coro, name: str, *, tracked_tasks: Optional[List[asyncio.Task]] = None
    ) -> None:
        if tracked_tasks is None:
            tracked_tasks = self._background_tasks
        task = asyncio.create_task(coro, name=name)
        tracked_tasks.append(task)

        def _cleanup_done(done_task: asyncio.Task) -> None:
            try:
                tracked_tasks.remove(done_task)
            except ValueError:
                pass

        task.add_done_callback(_cleanup_done)

    def _schedule_client_close(self, client: Any) -> None:
        if not hasattr(self, "_client_close_tasks"):
            self._client_close_tasks: List[asyncio.Task] = []

        async def _close() -> None:
            try:
                await client.close()
            except Exception:
                logger.debug("close stale tts client failed", exc_info=True)

        self._track_background_task(
            _close(), "close_stale_minimax_tts", tracked_tasks=self._client_close_tasks
        )

    async def _start_background_tasks(self) -> None:
        if self._cleanup_task_started:
            return
        self._cleanup_task_started = True
        self._track_background_task(self._periodic_audio_cleanup(), "audio_cleanup")
        self._track_background_task(self._periodic_session_cleanup(), "session_cleanup")

    async def _periodic_audio_cleanup(self) -> None:
        try:
            while True:
                await cleanup_dir(TEMP_DIR, ttl_seconds=AUDIO_CLEANUP_TTL_SECONDS)
                await asyncio.sleep(AUDIO_CLEANUP_TTL_SECONDS // 2)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("audio cleanup failed", exc_info=True)

    async def _periodic_session_cleanup(self) -> None:
        try:
            while True:
                await asyncio.sleep(SESSION_CLEANUP_INTERVAL_SECONDS)
                self._cleanup_stale_sessions()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("session cleanup failed", exc_info=True)

    def _cleanup_stale_sessions(self) -> None:
        now = time.time()
        stale_sessions = {
            sid
            for sid, state in self._session_state.items()
            if now - state.last_ts > SESSION_MAX_IDLE_SECONDS
        }

        if len(self._session_state) > SESSION_MAX_COUNT:
            keep = sorted(
                self._session_state.items(),
                key=lambda item: item[1].last_ts,
                reverse=True,
            )[:SESSION_MAX_COUNT]
            allowed_ids = {sid for sid, _ in keep}
            stale_sessions.update(
                sid for sid in self._session_state if sid not in allowed_ids
            )

        for sid in stale_sessions:
            self._session_state.pop(sid, None)
        self._cleanup_stale_inflight(now)

    def _build_inflight_sig(self, umo: str, text: str) -> str:
        digest = hashlib.sha1(f"{umo}:{text[:200]}".encode("utf-8")).hexdigest()
        return f"{umo}:{digest[:24]}"

    def _cleanup_stale_inflight(self, now: Optional[float] = None) -> None:
        if not self._inflight_sigs:
            return
        now_ts = now if now is not None else time.time()
        expired = [
            sig
            for sig, ts in self._inflight_sigs.items()
            if (now_ts - ts) > INFLIGHT_SIG_TTL_SECONDS
        ]
        for sig in expired:
            self._inflight_sigs.pop(sig, None)
        if len(self._inflight_sigs) > INFLIGHT_SIG_MAX_COUNT:
            oldest = sorted(self._inflight_sigs.items(), key=lambda item: item[1])
            for sig, _ in oldest[:-INFLIGHT_SIG_MAX_COUNT]:
                self._inflight_sigs.pop(sig, None)

    def _normalize_text(self, text: str) -> str:
        return self.marker_processor.normalize_text(text)

    def _strip_emo_head_many(self, text: str) -> Tuple[str, Optional[str]]:
        return self.marker_processor.strip_head_many(text)

    def _strip_any_visible_markers(self, text: str) -> str:
        return self.marker_processor.strip_all_visible_markers(text)

    def _prepare_text_for_tts(self, text: str) -> PreparedSpeechText:
        api_cfg = self.config.get_api_config()
        return self.text_sanitizer.prepare(
            text or "",
            provider=str(api_cfg.get("provider", "")),
            model=str(api_cfg.get("model", "")),
        )

    def _build_record_only_chain(
        self, original_chain: List, audio_paths: List[str]
    ) -> List:
        chain: List = [Record(file=path) for path in audio_paths]
        for comp in original_chain:
            if not isinstance(comp, Plain):
                chain.append(comp)
        return chain

    def _should_use_segmented_tts(self, sid: str, text: str) -> bool:
        return (
            self.segmented_tts_enabled
            and self.config.is_segmented_output_enabled_for_umo(sid)
            and self.segmented_tts_processor.should_use_segmented(
                text, self.segmented_min_chars
            )
        )

    def _build_minimax_guidance_instruction(self) -> str:
        expressive_supported = self._supports_minimax_expressive_tags()
        lines = [
            "以下规则只在回复预计会被转成语音时生效。",
            "可以用自然换行表示段落切换。",
            "如需停顿，只能在两段可发音文本之间插入 <#x#>，x 为秒数，可保留两位小数。",
        ]
        if expressive_supported:
            tags = "、".join(f"({tag})" for tag in MINIMAX_EXPRESSIVE_TAGS)
            lines.append(f"如自然需要，可少量使用这些语气标签：{tags}。不要堆砌。")
        else:
            lines.append("当前模型不要输出 MiniMax 语气标签，例如 (laughs)。")
        return "\n".join(lines)

    def _current_tts_model(self) -> str:
        api_cfg = self.config.get_api_config()
        return str(api_cfg.get("model", "") or "").strip().lower()

    def _supports_minimax_expressive_tags(self, model: Optional[str] = None) -> bool:
        current_model = str(model or self._current_tts_model() or "").strip().lower()
        return current_model in MINIMAX_EXPRESSIVE_MODELS

    def _publish_output_marker_mode(self, event: AstrMessageEvent) -> str:
        mode = (
            OUTPUT_MARKER_MODE_PRESERVE
            if self.config.is_voice_output_enabled_for_umo(self._get_umo(event))
            else OUTPUT_MARKER_MODE_STRIP
        )
        try:
            event.set_extra(OUTPUT_MARKER_MODE_EXTRA, mode)
        except Exception:
            logger.debug("set extra failed", exc_info=True)
        return mode

    async def _build_manual_tts_chain(
        self,
        event: AstrMessageEvent,
        text: str,
    ) -> Tuple[bool, List, str]:
        prepared = self._prepare_text_for_tts(text)
        tts_text = (prepared.tts_text or "").strip()
        if not tts_text:
            return False, [], "没有可用于语音合成的文本。"

        sid = self._get_umo(event)
        st = self._get_session_state(sid)
        segmented_enabled = self._should_use_segmented_tts(sid, tts_text)
        if segmented_enabled:
            seg_res = await self.segmented_tts_processor.process_only(tts_text, st)
            audio_paths = [
                self.tts_processor.normalize_audio_path(seg.audio_path)
                for seg in seg_res.successful_segments
                if seg.audio_path
            ]
            if audio_paths:
                return (
                    True,
                    [Record(file=path) for path in audio_paths],
                    prepared.display_text or tts_text,
                )
            return False, [], f"TTS 合成失败：{seg_res.error or '分段合成失败'}"

        proc_res = await self.tts_processor.process(tts_text, st)
        if not proc_res.success or not proc_res.audio_path:
            return False, [], f"TTS 合成失败：{proc_res.error or '未知错误'}"
        audio_path = self.tts_processor.normalize_audio_path(proc_res.audio_path)
        return True, [Record(file=audio_path)], prepared.display_text or tts_text

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, request):
        try:
            self._publish_output_marker_mode(event)
        except Exception:
            logger.error("on_llm_request failed", exc_info=True)

    @filter.on_llm_response(priority=1)
    async def on_llm_response(self, event: AstrMessageEvent, response: LLMResponse):
        _ = event, response
        return

    @filter.on_decorating_result(priority=999)
    async def _final_strip_markers(self, event: AstrMessageEvent):
        if not self.emo_marker_enable:
            return
        try:
            result = event.get_result()
            if not result or not hasattr(result, "chain"):
                return
            for comp in list(result.chain):
                if isinstance(comp, Plain) and getattr(comp, "text", None):
                    comp.text = self._strip_any_visible_markers(comp.text)
        except Exception:
            logger.error("final marker cleanup failed", exc_info=True)

    @filter.on_decorating_result(priority=-1000)
    async def on_decorating_result(self, event: AstrMessageEvent):
        await self._start_background_tasks()
        self._cleanup_stale_inflight()

        try:
            result = event.get_result()
            if not result:
                return
            try:
                is_llm_response = result.is_llm_result()
            except Exception:
                is_llm_response = (
                    getattr(result, "result_content_type", None)
                    == ResultContentType.LLM_RESULT
                )
            if not is_llm_response:
                return
            if not hasattr(result, "chain") or result.chain is None:
                result.chain = []
        except Exception:
            logger.warning("inspect llm result failed", exc_info=True)
            return

        cleaned_chain = []
        for comp in result.chain:
            if isinstance(comp, Plain) and getattr(comp, "text", None):
                text0 = self._normalize_text(comp.text)
                cleaned, _ = self._strip_emo_head_many(text0)
                cleaned = self._strip_any_visible_markers(cleaned)
                if cleaned:
                    cleaned_chain.append(Plain(text=cleaned))
            else:
                cleaned_chain.append(comp)
        result.chain = cleaned_chain

        text_parts = [
            c.text.strip()
            for c in result.chain
            if isinstance(c, Plain) and c.text.strip()
        ]
        if not text_parts:
            return

        sid = self._get_umo(event)
        if not self.config.is_voice_output_enabled_for_umo(sid):
            return

        prepared = self._prepare_text_for_tts(
            self._normalize_text(" ".join(text_parts))
        )
        tts_text = (prepared.tts_text or "").strip()
        if not tts_text:
            return

        st = self._get_session_state(sid)
        allowed_components = {"Plain", "At", "Reply", "Image", "Face"}
        has_non_plain = any(
            type(comp).__name__ not in allowed_components for comp in result.chain
        )
        check_res = self.condition_checker.check_all(
            tts_text,
            st,
            has_non_plain_elements=has_non_plain,
            enable_probability=False,
        )
        if not check_res.passed:
            return

        sig = self._build_inflight_sig(sid, tts_text)
        if sig in self._inflight_sigs:
            return
        self._inflight_sigs[sig] = time.time()

        try:
            segmented_enabled = self._should_use_segmented_tts(sid, tts_text)
            if segmented_enabled:
                seg_res = await self.segmented_tts_processor.process_only(tts_text, st)
                audio_paths = [
                    self.tts_processor.normalize_audio_path(seg.audio_path)
                    for seg in seg_res.successful_segments
                    if seg.audio_path
                ]
                if audio_paths:
                    result.chain = self._build_record_only_chain(
                        result.chain, audio_paths
                    )
                    return

            proc_res = await self.tts_processor.process(tts_text, st)
            if proc_res.success and proc_res.audio_path:
                audio_path = self.tts_processor.normalize_audio_path(
                    proc_res.audio_path
                )
                result.chain = self._build_record_only_chain(result.chain, [audio_path])
        finally:
            self._inflight_sigs.pop(sig, None)

    async def _switch_voice_output_for_current_umo(
        self,
        event: AstrMessageEvent,
        *,
        enable: bool,
    ) -> str:
        umo = self._get_umo(event)
        policy = self.config.get_feature_policy("voice_output")
        if enable and not bool(policy.get("enable", True)):
            await self.config.set_voice_output_enable_async(True)
            policy = self.config.get_feature_policy("voice_output")
        mode = policy.get("mode", "blacklist")

        if mode == "whitelist":
            if enable:
                await self.config.add_to_enabled_umos_async(umo)
            else:
                await self.config.remove_from_enabled_umos_async(umo)
        else:
            if enable:
                await self.config.remove_from_disabled_umos_async(umo)
            else:
                await self.config.add_to_disabled_umos_async(umo)

        self._update_components_from_config()
        return umo

    @filter.command("tts_on", priority=1)
    async def tts_on(self, event: AstrMessageEvent):
        yield event.plain_result(
            "本版本已禁用自动文字转语音；仍可通过 tts_speak 工具或 tts_say 指令发送语音。"
        )

    @filter.command("tts_off", priority=1)
    async def tts_off(self, event: AstrMessageEvent):
        umo = await self._switch_voice_output_for_current_umo(event, enable=False)
        yield event.plain_result(f"当前会话已关闭自动语音。UMO={umo}")

    @filter.command("tts_all_on", priority=1)
    async def tts_all_on(self, event: AstrMessageEvent):
        yield event.plain_result(
            "本版本已禁用自动文字转语音；仍可通过 tts_speak 工具或 tts_say 指令发送语音。"
        )

    @filter.command("tts_all_off", priority=1)
    async def tts_all_off(self, event: AstrMessageEvent):
        _ = event
        await self.config.set_voice_output_enable_async(False)
        self._update_components_from_config()
        yield event.plain_result("已关闭全局自动语音。")

    @filter.command("tts_status", priority=1)
    async def tts_status(self, event: AstrMessageEvent):
        umo = self._get_umo(event)
        api_cfg = self.config.get_api_config()
        voice_enabled = self.config.is_voice_output_enabled_for_umo(umo)
        segmented_enabled = self.config.is_segmented_output_enabled_for_umo(umo)
        msg = (
            f"服务商: MiniMax\n"
            f"UMO: {umo}\n"
            f"自动语音: {voice_enabled}\n"
            f"分段语音: {segmented_enabled}\n"
            f"模型: {api_cfg.get('model')}\n"
            f"默认语言: {api_cfg.get('language_boost') or '未指定'}\n"
            f"默认音色: {api_cfg.get('voice_id')}\n"
            f"最短字数: {self.text_min_limit}\n"
            f"最长字数: {self.text_limit}\n"
            f"冷却时间: {self.cooldown}s"
        )
        yield event.plain_result(msg)

    @filter.command("tts_say", priority=1)
    async def tts_say(self, event: AstrMessageEvent, *, text: Optional[str] = None):
        content = (text or DEFAULT_TEST_TEXT).strip() or DEFAULT_TEST_TEXT
        ok, chain, message = await self._build_manual_tts_chain(event, content)
        if not ok:
            yield event.plain_result(message)
            return
        yield event.chain_result(chain)

    if hasattr(filter, "llm_tool"):

        @filter.llm_tool(name="tts_speak")
        async def tts_speak(self, event: AstrMessageEvent, text: str):
            """让大模型主动发送语音消息，长内容按配置逐段发送。

            普通回复保持文字，不会自动转成语音，也不要求每轮使用语音。
            结合当前交流、已知状态和用户意愿，仅在决定发送语音时调用 `tts_speak`。
            用户要求打字或不方便听时使用文字。
            参数 `text` 是实际说给对方听的话，不把这条语音描述成正在打字。
            语音的停顿或语气标签只用于工具参数，不用于普通文字回复。
            长内容按插件配置分段发送；根据工具结果判断是否送达，成功后不重复发送同一段文字。

            Args:
                text(string): 需要转成语音并发送的文本。

            Returns:
                string: 发送结果说明。
            """
            content = (text or "").strip()
            if not content:
                yield "文本为空"
                return

            prepared = self._prepare_text_for_tts(content)
            tts_text = (prepared.tts_text or "").strip()
            if not tts_text:
                yield "没有可用于语音合成的文本。"
                return

            sid = self._get_umo(event)
            st = self._get_session_state(sid)

            if self._should_use_segmented_tts(sid, tts_text):

                async def _send_segment(audio_path) -> bool:
                    await event.send(event.chain_result([Record(file=str(audio_path))]))
                    return True

                seg_res = await self.segmented_tts_processor.process_and_send(
                    tts_text,
                    st,
                    _send_segment,
                )
                successful_count = len(seg_res.successful_segments)
                if not successful_count:
                    yield f"TTS 合成失败：{seg_res.error or '分段发送失败'}"
                    return

                failed_count = len(seg_res.segments) - successful_count
                if failed_count:
                    logger.warning(
                        "tts_speak segmented send partially failed: sent=%s failed=%s",
                        successful_count,
                        failed_count,
                    )
            else:
                ok, chain, error_message = await self._build_manual_tts_chain(
                    event, content
                )
                if not ok:
                    yield error_message
                    return

                try:
                    await event.send(event.chain_result(chain))
                except Exception as exc:
                    logger.error("tts_speak send failed: %s", exc)
                    yield f"发送失败：{exc}"
                    return

            if hasattr(event, "clear_result"):
                try:
                    event.clear_result()
                except Exception:
                    logger.debug("clear result failed after tts_speak", exc_info=True)

            yield None
            return

    @filter.command("tts_segment_on", priority=1)
    async def tts_segment_on(self, event: AstrMessageEvent):
        umo = self._get_umo(event)
        policy = self.config.get_feature_policy("segmented_output")
        if not bool(policy.get("enable", False)):
            await self.config.set_feature_policy_async(
                "segmented_output", {"enable": True}
            )
            policy = self.config.get_feature_policy("segmented_output")
        if policy.get("mode") == "whitelist":
            await self.config.add_umo_to_feature(
                "segmented_output", umo, "enabled_umos"
            )
        else:
            await self.config.remove_umo_from_feature(
                "segmented_output", umo, "disabled_umos"
            )
        self._update_components_from_config()
        yield event.plain_result(f"当前会话已开启分段语音。UMO={umo}")

    @filter.command("tts_segment_off", priority=1)
    async def tts_segment_off(self, event: AstrMessageEvent):
        umo = self._get_umo(event)
        policy = self.config.get_feature_policy("segmented_output")
        if not bool(policy.get("enable", False)):
            await self.config.set_feature_policy_async(
                "segmented_output", {"enable": True}
            )
            policy = self.config.get_feature_policy("segmented_output")
        if policy.get("mode") == "whitelist":
            await self.config.remove_umo_from_feature(
                "segmented_output", umo, "enabled_umos"
            )
        else:
            await self.config.add_umo_to_feature(
                "segmented_output", umo, "disabled_umos"
            )
        self._update_components_from_config()
        yield event.plain_result(f"当前会话已关闭分段语音。UMO={umo}")
