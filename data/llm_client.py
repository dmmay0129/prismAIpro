"""统一 LLM 客户端：优先 DeepSeek API，回退本地 Ollama。

provider 配置（config/settings.json → content 段，或环境变量）：
- llm_provider: "auto"（默认）｜"deepseek"｜"ollama"
  auto = 配置了 deepseek_api_key 就用 DeepSeek，否则用本地 Ollama；
- deepseek_api_key: DeepSeek 开放平台 API Key（sk-...）；
- deepseek_model: 默认 deepseek-chat；
- deepseek_base_url: OpenAI 兼容的 chat/completions 地址。

DeepSeek 走 OpenAI 兼容的对话接口，回答质量远高于本地小模型，
追问更自然、评分更准；任何失败都逐级回退（DeepSeek → Ollama → 规则），
保证离线演示仍可跑通。
"""
from __future__ import annotations

import logging

from core.settings import get_settings

logger = logging.getLogger(__name__)


def _cfg() -> dict:
    return get_settings()["content"]


def _has_deepseek() -> bool:
    cfg = _cfg()
    return bool(str(cfg.get("deepseek_api_key") or "").strip())


def _provider() -> str:
    cfg = _cfg()
    provider = str(cfg.get("llm_provider") or "auto").strip().lower()
    if provider == "auto":
        return "deepseek" if _has_deepseek() else "ollama"
    if provider == "deepseek" and not _has_deepseek():
        logger.warning("llm_provider=deepseek 但未配置 deepseek_api_key，回退 Ollama")
        return "ollama"
    return provider


def active_provider() -> str:
    """当前实际使用的模型来源："deepseek" 或 "ollama"。"""
    return _provider()


def chat_text(system: str, user: str, json_mode: bool = False) -> str | None:
    """发送一轮对话，返回模型输出文本；所有途径都失败时返回 None。

    json_mode=True 时要求模型只输出 JSON（DeepSeek 用 response_format，
    Ollama 用 format 参数）；解析仍由调用方自行兜底。
    """
    cfg = _cfg()
    timeout = float(cfg.get("timeout_sec") or 30)
    provider = _provider()

    if provider == "deepseek":
        try:
            import requests

            payload = {
                "model": str(cfg.get("deepseek_model") or "deepseek-chat"),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "max_tokens": int(cfg.get("deepseek_max_tokens") or 2048),
                "temperature": 0.3,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            resp = requests.post(
                str(cfg.get("deepseek_base_url") or "https://api.deepseek.com/chat/completions"),
                json=payload,
                headers={
                    "Authorization": "Bearer " + str(cfg["deepseek_api_key"]).strip(),
                    "Content-Type": "application/json",
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            text = str(resp.json()["choices"][0]["message"]["content"] or "").strip()
            if text:
                return text
            logger.info("DeepSeek 返回空内容，回退本地模型")
        except (requests.RequestException, ValueError, TypeError, KeyError, IndexError) as exc:
            logger.info("DeepSeek 不可用，回退本地 Ollama: %s", exc)
    # 本地 Ollama（/api/generate，prompt 模式）
    try:
        import requests

        resp = requests.post(
            cfg["ollama_url"],
            json={
                "model": cfg["model"],
                "prompt": f"{system}\n{user}",
                "stream": False,
                **({"format": "json"} if json_mode else {}),
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        text = str(resp.json().get("response", "") or "").strip()
        return text or None
    except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
        logger.info("本地 Ollama 不可用: %s", exc)
        return None
