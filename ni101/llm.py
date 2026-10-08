"""LLM 客户端：OpenAI 兼容协议 + 未配置时的「不可用」降级。

降级原则
--------
未配置 api_key / base_url 时，本模块**不会编造内容**，而是返回
``ok=False, source="unavailable"``。上层（Extractor / Critic）据此切换到
确定性规则实现——规则实现是从 PDF 里真实读出来的结果，不是虚构的。

这样做的代价是"安静地走了降级分支"，所以必须配 ``scripts/check_llm.py``
自检，把失败点暴露出来。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from .config import ConfigError, LLMSpec, assert_ascii


class LLMResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    ok: bool = False
    text: str = ""
    error: Optional[str] = None
    latency_ms: int = 0
    source: str = "llm"           # llm | unavailable
    model: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


# --------------------------------------------------------------------------
# JSON 抽取
# --------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """去掉 ```json ... ``` 包裹。"""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _balanced_slice(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    """从首个 open_ch 起找到配对的 close_ch（跳过字符串内的括号）。"""
    start = text.find(open_ch)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == open_ch:
            depth += 1
        elif char == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json(text: str) -> Any:
    """从模型输出里尽力解析 JSON。

    依次尝试：整体解析 -> 去代码围栏 -> 配对大括号 -> 配对方括号。
    全部失败抛 ``ValueError``，由调用方记为该轮失败。
    """
    if not text or not text.strip():
        raise ValueError("模型返回为空")

    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)
    candidates.append(strip_code_fence(stripped))
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        chunk = _balanced_slice(stripped, open_ch, close_ch)
        if chunk:
            candidates.append(chunk)

    last_error: Optional[Exception] = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    raise ValueError(f"无法从模型输出中解析出 JSON：{last_error}") from last_error


# --------------------------------------------------------------------------
# 客户端
# --------------------------------------------------------------------------
class BaseLLM:
    """LLM 统一接口。"""

    name: str = "llm"

    def __init__(self, spec: LLMSpec, retries: int = 2) -> None:
        self.spec = spec
        self.retries = max(0, retries)
        self.name = spec.label or "llm"

    @property
    def available(self) -> bool:
        raise NotImplementedError

    def complete(self, system: str, user: str, **kwargs: Any) -> LLMResult:
        raise NotImplementedError

    def complete_json(self, system: str, user: str, **kwargs: Any) -> tuple[Any, LLMResult]:
        """返回 (解析结果, 原始结果)。解析失败时抛 ValueError。"""
        result = self.complete(system, user, **kwargs)
        if not result.ok:
            raise ValueError(result.error or "LLM 调用失败")
        return extract_json(result.text), result


class UnavailableLLM(BaseLLM):
    """未配置端点时的占位实现——显式失败，绝不编造。"""

    @property
    def available(self) -> bool:
        return False

    def complete(self, system: str, user: str, **kwargs: Any) -> LLMResult:
        return LLMResult(
            ok=False,
            source="unavailable",
            model=self.spec.model,
            error=(
                f"[{self.name}] 未配置 api_key / base_url，已跳过真实模型调用。"
                "请在 config.yaml 的「填写点」处补全，或设置对应环境变量。"
            ),
        )


class OpenAICompatLLM(BaseLLM):
    """OpenAI 兼容客户端。DeepSeek / Qwen / GLM / OpenAI / Ollama 通用。"""

    def __init__(self, spec: LLMSpec, retries: int = 2, verbose: bool = False) -> None:
        super().__init__(spec, retries=retries)
        self.verbose = verbose
        self._client = None

    @property
    def available(self) -> bool:
        return self.spec.configured

    def _build_client(self) -> Any:
        if self._client is not None:
            return self._client
        # 进 HTTP 头的值必须 ASCII，先判再发，避免看不懂的 UnicodeEncodeError
        api_key = assert_ascii(self.spec.api_key.strip(), f"{self.name}.api_key")
        base_url = assert_ascii(self.spec.base_url.strip(), f"{self.name}.base_url")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ConfigError(
                "未安装 openai 包。请执行：pip install openai>=1.30"
            ) from exc
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.spec.timeout_s,
            max_retries=0,  # 重试由本模块自己控制，便于记录每次尝试
        )
        return self._client

    def complete(self, system: str, user: str, **kwargs: Any) -> LLMResult:
        if not self.available:
            return UnavailableLLM(self.spec).complete(system, user)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload = {
            "model": self.spec.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.spec.temperature),
            "max_tokens": kwargs.get("max_tokens", self.spec.max_tokens),
        }
        if kwargs.get("json_mode"):
            payload["response_format"] = {"type": "json_object"}

        last_error = ""
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            try:
                client = self._build_client()
                response = client.chat.completions.create(**payload)
                latency_ms = int((time.perf_counter() - started) * 1000)
                content = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                return LLMResult(
                    ok=True,
                    text=content,
                    latency_ms=latency_ms,
                    source="llm",
                    model=self.spec.model,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                )
            except ConfigError:
                raise
            except Exception as exc:  # noqa: BLE001 - 要的是可读原因而不是堆栈
                last_error = f"{type(exc).__name__}: {exc}"
                if self.verbose:
                    print(f"[llm:{self.name}] 第 {attempt + 1} 次调用失败：{last_error}")
                if attempt < self.retries:
                    time.sleep(min(2.0 * (attempt + 1), 6.0))

        return LLMResult(
            ok=False,
            error=f"[{self.name}] 连续 {self.retries + 1} 次调用失败：{last_error}",
            source="llm",
            model=self.spec.model,
        )

    def probe(self, quick: bool = False) -> LLMResult:
        """连通性自检：发一次最小请求（打的是真实会被调用的端点）。"""
        if not self.available:
            return LLMResult(
                ok=False,
                source="unavailable",
                error=f"[{self.name}] 未配置，跳过探测",
            )
        if quick:
            return self.complete("You are a probe.", "ping", max_tokens=1)
        return self.complete(
            "You are a probe. Reply with exactly: OK",
            "Say OK.",
            max_tokens=8,
        )


def build_llm(spec: LLMSpec, retries: int = 2, verbose: bool = False) -> BaseLLM:
    """按配置返回可用客户端；未配置则返回显式不可用实现。"""
    if spec.configured:
        return OpenAICompatLLM(spec, retries=retries, verbose=verbose)
    return UnavailableLLM(spec)
