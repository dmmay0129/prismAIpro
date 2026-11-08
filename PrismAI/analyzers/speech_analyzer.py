"""语音分析过滤器（策略模式具体策略之一）。

优先使用 faster-whisper 转写 + librosa 提取特征；
依赖缺失或音频不可读时自动降级为规则估计，保证流水线不中断（第4周异常分支）。
"""
from __future__ import annotations

import logging
import os
import threading
import wave
from typing import Any, Dict, Tuple

from core.base_analyzer import BaseAnalyzer
from core.schemas import AnalysisInput, AnalyzerResult
from core.settings import get_settings

logger = logging.getLogger(__name__)

FILLERS = ["嗯", "呃", "额", "那个", "就是", "然后然后", "对对对"]
# 正常语速区间（字/分）：只要不低于 50、不高于 450 都算正常，
# 区间内不扣分、不提示；语速只是辅助观察项，不是评判重点。
NORMAL_RATE = (50, 450)

# Whisper 模型加载成本高，进程内复用；Streamlit 重跑不会重复加载。
_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}
_MODEL_LOCK = threading.Lock()

# 繁体转简体：whisper 对 zh 的转写常输出繁体，统一转简体后展示与评分。
_T2S = None


def _to_simplified(text: str) -> str:
    global _T2S
    if _T2S is None:
        try:
            from opencc import OpenCC

            _T2S = OpenCC("t2s")
        except Exception:
            logger.warning("OpenCC 不可用，转写保持原样")
            _T2S = False
    if _T2S is False:
        return text
    try:
        return str(_T2S.convert(text))
    except Exception:
        return text


def _wav_duration(path: str) -> float:
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        size = os.path.getsize(path)
        return max(size / 32000.0, 1.0)  # 粗略估计：16kHz 16bit 单声道


def _get_model():
    """按配置加载并缓存 WhisperModel（线程安全）。"""
    cfg = get_settings()["speech"]
    key = (str(cfg["model_size"]), str(cfg["device"]), str(cfg["compute_type"]))
    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            from faster_whisper import WhisperModel

            model = WhisperModel(key[0], device=key[1], compute_type=key[2])
            _MODEL_CACHE[key] = model
            logger.info("Whisper 模型已加载: %s/%s/%s", *key)
        return model


def _transcribe(path: str) -> Tuple[str | None, bool]:
    """faster-whisper 转写，返回 (文本, 是否无有效语音)。

    - (None, False)：模型/依赖不可用或音频无法解码 → 降级评估；
    - (None, True)：音频存在、模型正常，但未检出任何有效语音
      （静音、纯噪声/哔声，或 VAD 过滤后无片段）→ 无效作答；
    - (文本, False)：正常转写。
    """
    try:
        cfg = get_settings()["speech"]
        model = _get_model()
        segments, _ = model.transcribe(
            path,
            language=cfg["language"],
            beam_size=int(cfg["beam_size"]),
            vad_filter=bool(cfg["vad_filter"]),
        )
        segs = list(segments)
        if not segs:
            return None, True
        text = "".join(s.text for s in segs).strip()
        return (text or None), (not text)
    except Exception as exc:
        logger.warning("语音转写失败，将降级: %s", exc)
        return None, False


class SpeechAnalyzer(BaseAnalyzer):
    modality = "speech"

    def _run(self, inp: AnalysisInput) -> AnalyzerResult:
        if not inp.audio_path or not os.path.exists(inp.audio_path):
            return AnalyzerResult(
                modality=self.modality, status="degraded", score=60.0,
                suggestions=["未检测到音频输入，语音维度按基线分处理"],
            )

        duration = _wav_duration(inp.audio_path)
        transcript, no_speech = _transcribe(inp.audio_path)

        if transcript is None:
            if no_speech:
                # 音频存在但没有任何有效语音：静音/噪声不是回答，
                # 语音维度给底线分（不给 65 的降级基线分），并打标
                # 供内容维度判无效作答（综合分归零）。
                return AnalyzerResult(
                    modality=self.modality, status="ok", score=20.0,
                    metrics={
                        "duration_sec": round(duration, 1),
                        "no_speech": True,
                    },
                    suggestions=["未检测到有效语音内容（静音或非语音声音），本轮按无效作答处理"],
                )
            # 降级：仅基于时长估计（模型/依赖不可用，属系统问题）
            return AnalyzerResult(
                modality=self.modality, status="degraded", score=65.0,
                metrics={"duration_sec": round(duration, 1)},
                suggestions=[
                    "语音转写不可用（模型加载失败或音频无法解码），本次为降级评估"
                ],
            )

        transcript = _to_simplified(transcript)
        minutes = max(duration / 60.0, 1e-6)
        n_chars = len(transcript)
        rate = n_chars / minutes
        filler_hits = sum(transcript.count(f) for f in FILLERS)
        filler_per_min = filler_hits / minutes

        # 语速评分：正常区间内满分；只有明显过快/过慢才温和扣分。
        # 语速不是评判重点，权重弱化（填充词更能反映表达流畅度）。
        if NORMAL_RATE[0] <= rate <= NORMAL_RATE[1]:
            rate_score = 100.0
        else:
            dev = min(abs(rate - NORMAL_RATE[0]), abs(rate - NORMAL_RATE[1]))
            rate_score = max(100.0 - dev * 0.4, 65.0)
        filler_score = max(100.0 - filler_per_min * 12, 30.0)
        score = round(rate_score * 0.25 + filler_score * 0.75, 1)

        suggestions = []
        if rate > NORMAL_RATE[1]:
            suggestions.append(
                f"语速 {rate:.0f} 字/分偏快，建议适当放慢"
                f"（正常约 {NORMAL_RATE[0]}-{NORMAL_RATE[1]} 字/分）"
            )
        elif rate < NORMAL_RATE[0]:
            suggestions.append(
                f"语速 {rate:.0f} 字/分偏慢，建议适当加快"
                f"（正常约 {NORMAL_RATE[0]}-{NORMAL_RATE[1]} 字/分）"
            )
        if filler_per_min > 2:
            suggestions.append(f"填充词约 {filler_per_min:.1f} 次/分（{filler_hits} 次），建议减少口头禅")

        return AnalyzerResult(
            modality=self.modality, status="ok", score=score,
            metrics={
                "transcript": transcript,
                "duration_sec": round(duration, 1),
                "speech_rate": round(rate, 1),
                "filler_count": filler_hits,
                "model": get_settings()["speech"]["model_size"],
            },
            suggestions=suggestions or ["语音表达节奏良好"],
        )
