"""pytest 公共夹具。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ni101.config import AppConfig, LLMSpec, PipelineSettings, load_config
from ni101.llm import BaseLLM, LLMResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def demo_pdf_dir(project_root: Path) -> Path:
    pdf_dir = project_root / "data" / "pdfs"
    if not list(pdf_dir.glob("*.pdf")):
        pytest.skip("演示数据缺失，请先运行 python scripts/make_demo_data.py")
    return pdf_dir


@pytest.fixture()
def config() -> AppConfig:
    """测试用配置：模型端点留空，强制走确定性规则路径。"""
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    cfg.extractor.api_key = ""
    cfg.extractor.base_url = ""
    cfg.critic.api_key = ""
    cfg.critic.base_url = ""
    return cfg


@pytest.fixture()
def settings() -> PipelineSettings:
    return PipelineSettings(
        max_rounds=3,
        pass_score=8.0,
        value_tolerance=0.05,
        consistency_tolerance=0.15,
        consistency_hard_fail=0.35,
        min_field_accuracy_to_pass=0.80,
        few_shot_limit=3,
    )


class ScriptedLLM(BaseLLM):
    """按脚本返回固定 JSON 的假模型，用于测真实的 LLM 路径（不联网）。

    第 1 次调用返回 responses[0]，第 2 次 responses[1]，之后一直复用最后一个。
    """

    def __init__(self, responses: list[object]) -> None:
        spec = LLMSpec(
            label="scripted",
            api_key="test-key",
            base_url="http://127.0.0.1:1/v1",
            model="scripted-model",
        )
        super().__init__(spec, retries=0)
        self._responses = list(responses)
        self.calls = 0
        self.prompts: list[str] = []

    @property
    def available(self) -> bool:
        return True

    def complete(self, system: str, user: str, **kwargs: object) -> LLMResult:
        self.prompts.append(user)
        index = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        payload = self._responses[index]
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return LLMResult(ok=True, text=text, model="scripted-model", latency_ms=1)


@pytest.fixture()
def scripted_llm_factory():
    def _make(responses: list[object]) -> ScriptedLLM:
        return ScriptedLLM(responses)

    return _make
