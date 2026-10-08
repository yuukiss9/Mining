"""配置加载与校验。

这个模块承担一个容易被忽略的职责：**证明配置真的生效了**。

降级设计有一个副作用——配置写错时系统不会报错，而是安静地走规则分支。
所以这里提供 :func:`describe` 输出脱敏后的实际取值，配合 ``scripts/check_llm.py``
可以一眼看出「主人以为在用真模型，其实没跑」的情况。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

#: 环境变量前缀：NI101_<LABEL>_<FIELD>
ENV_PREFIX = "NI101"


class ConfigError(RuntimeError):
    """配置层可读错误——要求能直接照着改，不要抛出难懂的堆栈。"""


def assert_ascii(value: str, field_name: str) -> str:
    """进 HTTP 头的值必须 ASCII。

    API Key 从文档/网页粘贴时极易混入全角字符或中文占位符，
    一旦放进 header 就抛 ``UnicodeEncodeError``，报错信息完全看不出是 Key 的问题。
    在构造请求前先判，给出可读原因。
    """
    if not value:
        return value
    try:
        value.encode("latin-1")
    except UnicodeEncodeError as exc:
        bad = "".join(ch for ch in value if ord(ch) > 127)
        raise ConfigError(
            f"{field_name} 含有非 ASCII 字符 {bad!r}。"
            "通常是粘贴时带入了全角字符或中文占位符，请重新粘贴纯 ASCII 的取值。"
        ) from exc
    return value


class LLMSpec(BaseModel):
    """一个 Agent 所使用的模型端点。api_key / base_url 留空即进入降级模式。"""

    model_config = ConfigDict(protected_namespaces=())

    label: str = ""
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_s: int = 180

    @property
    def configured(self) -> bool:
        """三项齐全才算配置好；缺任意一项都按未配置处理（避免半配置的静默失败）。"""
        return bool(self.api_key.strip() and self.base_url.strip() and self.model.strip())

    @property
    def missing(self) -> list[str]:
        gaps = []
        if not self.api_key.strip():
            gaps.append("api_key")
        if not self.base_url.strip():
            gaps.append("base_url")
        if not self.model.strip():
            gaps.append("model")
        return gaps

    def masked_key(self) -> str:
        key = self.api_key.strip()
        if not key:
            return "<empty>"
        if len(key) <= 8:
            return key[:2] + "*" * (len(key) - 2)
        return f"{key[:4]}...{key[-4:]} (len={len(key)})"

    def describe(self) -> str:
        if not self.configured:
            return (
                f"{self.label or 'llm'}: 未配置（缺少 {', '.join(self.missing)}）"
                f" -> 降级为规则模式"
            )
        return (
            f"{self.label or 'llm'}: model={self.model} base_url={self.base_url} "
            f"key={self.masked_key()}"
        )


class PipelineSettings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    max_rounds: int = 3
    pass_score: float = 8.0
    value_tolerance: float = 0.05
    consistency_tolerance: float = 0.15
    consistency_hard_fail: float = 0.35
    min_field_accuracy_to_pass: float = 0.80
    few_shot_limit: int = 3


class PathSettings(BaseModel):
    pdf_dir: str = "data/pdfs"
    ground_truth_dir: str = "data/ground_truth"
    output_dir: str = "outputs"


class RuntimeSettings(BaseModel):
    log_level: str = "info"
    request_retries: int = 2


class AppConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    extractor: LLMSpec = Field(default_factory=lambda: LLMSpec(label="extractor"))
    critic: LLMSpec = Field(default_factory=lambda: LLMSpec(label="critic"))
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    config_path: Path = DEFAULT_CONFIG_PATH

    # -- 路径解析 --------------------------------------------------------
    @property
    def pdf_dir(self) -> Path:
        return self.resolve(self.paths.pdf_dir)

    @property
    def ground_truth_dir(self) -> Path:
        return self.resolve(self.paths.ground_truth_dir)

    @property
    def output_dir(self) -> Path:
        return self.resolve(self.paths.output_dir)

    @property
    def extraction_dir(self) -> Path:
        return self.output_dir / "extractions"

    @property
    def evolution_log(self) -> Path:
        return self.output_dir / "evolution.jsonl"

    def resolve(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else (PROJECT_ROOT / path)

    def describe(self) -> str:
        return "\n".join(
            [
                self.extractor.describe(),
                self.critic.describe(),
                (
                    f"pipeline: max_rounds={self.pipeline.max_rounds} "
                    f"pass_score={self.pipeline.pass_score} "
                    f"value_tolerance=±{self.pipeline.value_tolerance:.0%}"
                ),
            ]
        )


def _env_override(spec: LLMSpec, key_name: str) -> None:
    """用环境变量覆盖配置项（优先级高于 yaml），便于不把 key 落到文件里。"""
    label = (spec.label or "").upper()
    if not label:
        return
    env_key = f"{ENV_PREFIX}_{label}_{key_name.upper()}"
    if env_key in os.environ and os.environ[env_key].strip():
        setattr(spec, key_name, os.environ[env_key].strip())


def load_config(path: Optional[Path] = None) -> AppConfig:
    """加载 config.yaml，套用环境变量覆盖，并做一次取值自检。

    文件不存在时不报错——直接用默认值（全降级），保证工程在任何环境下都能启动。
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if config_path.exists():
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"{config_path} 不是合法 YAML：{exc}\n"
                "常见原因：值里含未加引号的特殊字符（如 : #），或缩进用了 Tab。"
            ) from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{config_path} 顶层必须是映射（key: value），当前是 {type(loaded).__name__}")
        raw = loaded

    config = AppConfig(**raw)
    config.config_path = config_path

    for spec, label in ((config.extractor, "extractor"), (config.critic, "critic")):
        spec.label = spec.label or label
        for field in ("api_key", "base_url", "model"):
            _env_override(spec, field)
        # 只对非空值做 ASCII 校验：空值是"未配置"，不是错误
        assert_ascii(spec.api_key.strip(), f"{spec.label}.api_key")
        assert_ascii(spec.base_url.strip(), f"{spec.label}.base_url")

    return config


def ensure_directories(config: AppConfig) -> None:
    for path in (
        config.pdf_dir,
        config.ground_truth_dir,
        config.output_dir,
        config.extraction_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
