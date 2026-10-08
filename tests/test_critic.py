"""CriticMaster Agent 的单测：确定性规则层必须能抓出各类"明显错误"。"""

from __future__ import annotations

from pathlib import Path

import pytest

from ni101.critic import CriticMasterAgent
from ni101.llm import UnavailableLLM
from ni101.config import LLMSpec
from ni101.pdf import PdfDocument
from ni101.schema import ResourceRow

PAGE_TEXT = (
    "Mineral Resource Statement. Indicated Mineral Resources are reported "
    "above a cut-off grade of 0.50 g/t Au."
)

#: 故意提到 Inferred，用于验证"原文有、结果里没有"的覆盖缺口检查
PAGE_TEXT_WITH_INFERRED = PAGE_TEXT + " Inferred Mineral Resources are also reported."


def make_doc(page_text: str = PAGE_TEXT) -> PdfDocument:
    return PdfDocument(path=Path("synthetic.pdf"), page_count=1, page_texts=[page_text])


def row(**kwargs) -> ResourceRow:
    base = dict(
        resource_class="indicated",
        commodity="Au",
        tonnage_mt=45.8,
        grade=1.21,
        grade_unit="g/t",
        contained_metal=1782,
        contained_metal_unit="koz",
        source_page=1,
        source_snippet="Indicated 45.8 1.21 1,782",
    )
    base.update(kwargs)
    return ResourceRow(**base)


@pytest.fixture()
def critic(settings) -> CriticMasterAgent:
    return CriticMasterAgent(UnavailableLLM(LLMSpec(label="critic")), settings)


class TestCleanRows:
    def test_clean_rows_score_full(self, critic: CriticMasterAgent):
        verdict = critic.review(make_doc(), [row()])
        assert verdict.score == 10.0
        assert verdict.verdict == "pass"
        assert not verdict.issues
        assert verdict.source == "rule"

    def test_multiple_clean_rows(self, critic: CriticMasterAgent):
        rows = [
            row(),
            row(resource_class="inferred", tonnage_mt=23.6, grade=0.94, contained_metal=713),
        ]
        verdict = critic.review(make_doc(), rows)
        assert verdict.score == 10.0
        assert verdict.verdict == "pass"


class TestConsistencyViolation:
    def test_hard_violation_triggers_row_abstain(self, critic: CriticMasterAgent):
        rows = [
            row(resource_class="indicated", tonnage_mt=64.3, grade=1.18, grade_unit="%",
                contained_metal=0.76, contained_metal_unit="Mt", commodity="Li2O"),
            row(resource_class="inferred", tonnage_mt=32.7, grade=1.05, grade_unit="%",
                contained_metal=0.13, contained_metal_unit="Mt", commodity="Li2O"),
        ]
        verdict = critic.review(make_doc(), rows)
        assert verdict.abstain_row_indices == [1]
        assert not verdict.hard_abstain  # 只有一行坏，不该全篇拒答
        # 拒答行退出"承诺正确"的范围，剩余行仍是满分
        assert verdict.score == 10.0
        critical = [i for i in verdict.issues if i.severity == "critical"]
        assert len(critical) == 1
        assert critical[0].require_abstain is True
        assert "62%" in critical[0].message or "62." in critical[0].message

    def test_all_rows_bad_triggers_document_abstain(self, critic: CriticMasterAgent):
        rows = [
            row(tonnage_mt=10.0, grade=1.0, contained_metal=999, contained_metal_unit="koz"),
        ]
        verdict = critic.review(make_doc(), rows)
        # 10 Mt x 1.0 g/t = 10 t = 321.5 koz，印刷 999 koz -> 偏差 210%
        assert verdict.hard_abstain is True
        assert verdict.verdict == "abstain"
        assert verdict.score <= 3.0

    def test_moderate_deviation_is_critical_but_not_abstain(self, critic: CriticMasterAgent):
        # 45.8 Mt @ 1.21 g/t = 55.418 t = 1781.6 koz；印刷 1490 koz -> 偏差约 16.4%
        verdict = critic.review(make_doc(), [row(contained_metal=1490)])
        assert not verdict.abstain_row_indices
        assert any(i.severity == "critical" for i in verdict.issues)
        assert verdict.score < 8.0


class TestFieldChecks:
    def test_missing_fields_are_major(self, critic: CriticMasterAgent):
        verdict = critic.review(make_doc(), [row(contained_metal=None, contained_metal_unit=None)])
        types = {i.issue_type for i in verdict.issues}
        assert "missing_field" in types
        assert verdict.score == pytest.approx(7.0)  # 10 - 1.5 x 2
        assert verdict.verdict == "revise"

    def test_percent_over_100_is_unit_confusion(self, critic: CriticMasterAgent):
        verdict = critic.review(make_doc(), [row(grade=118.0, grade_unit="%")])
        issues = [i for i in verdict.issues if i.issue_type == "unit_confusion"]
        assert issues and issues[0].severity == "critical"

    def test_implausible_tonnage_magnitude(self, critic: CriticMasterAgent):
        verdict = critic.review(make_doc(), [row(tonnage_mt=5_000_000.0)])
        assert any(i.issue_type == "implausible_magnitude" for i in verdict.issues)

    def test_missing_provenance_is_minor(self, critic: CriticMasterAgent):
        verdict = critic.review(make_doc(), [row(source_page=None, source_snippet=None)])
        minors = [i for i in verdict.issues if i.severity == "minor"]
        assert len(minors) == 2
        assert verdict.score == pytest.approx(10.0 - 0.8)
        assert verdict.verdict == "pass"  # 只有 minor 不该卡住


class TestDocumentChecks:
    def test_coverage_gap_detected(self, critic: CriticMasterAgent):
        # 原文提到 Inferred 却只抽到 Indicated
        verdict = critic.review(make_doc(PAGE_TEXT_WITH_INFERRED), [row()])
        assert any(i.issue_type == "coverage_gap" for i in verdict.issues)

    def test_duplicate_rows_detected(self, critic: CriticMasterAgent):
        verdict = critic.review(make_doc(), [row(), row()])
        assert any(i.issue_type == "duplicate_row" for i in verdict.issues)

    def test_unclassified_row_is_critical(self, critic: CriticMasterAgent):
        verdict = critic.review(make_doc(), [row(resource_class=None)])
        assert any(i.issue_type == "unclassified_row" for i in verdict.issues)

    def test_empty_rows_is_hard_abstain(self, critic: CriticMasterAgent):
        verdict = critic.review(make_doc(), [])
        assert verdict.hard_abstain is True
        assert verdict.verdict == "abstain"
        assert any(i.issue_type == "no_rows" for i in verdict.issues)


class TestLlmLayer:
    def test_llm_score_cannot_override_rule_evidence(self, settings, scripted_llm_factory):
        """规则层发现硬矛盾时，LLM 给高分也不能盖过去。"""
        llm = scripted_llm_factory(
            [{"score": 10.0, "verdict": "pass", "summary": "看起来没问题", "issues": []}]
        )
        critic = CriticMasterAgent(llm, settings)
        rows = [row(tonnage_mt=10.0, grade=1.0, contained_metal=999, contained_metal_unit="koz")]
        verdict = critic.review(make_doc(), rows)
        assert verdict.score <= 3.0
        assert verdict.verdict == "abstain"
        assert verdict.source == "llm+rule"

    def test_llm_abstain_is_respected(self, settings, scripted_llm_factory):
        llm = scripted_llm_factory(
            [{"score": 3.0, "verdict": "abstain", "summary": "表格残缺",
              "abstain_rows": [], "issues": []}]
        )
        critic = CriticMasterAgent(llm, settings)
        verdict = critic.review(make_doc(), [row()])
        assert verdict.hard_abstain is True
        assert verdict.verdict == "abstain"

    def test_llm_abstain_rows_are_merged(self, settings, scripted_llm_factory):
        llm = scripted_llm_factory(
            [{"score": 7.0, "verdict": "revise", "summary": "第二行可疑",
              "abstain_rows": [1],
              "issues": [{"row_index": 1, "field": "grade", "severity": "critical",
                          "issue_type": "llm_finding", "message": "与原文不符"}]}]
        )
        critic = CriticMasterAgent(llm, settings)
        rows = [row(), row(resource_class="inferred", tonnage_mt=23.6, grade=0.94,
                           contained_metal=713)]
        verdict = critic.review(make_doc(), rows)
        assert 1 in verdict.abstain_row_indices
        assert verdict.score == pytest.approx(7.0)

    def test_llm_failure_falls_back_to_rule(self, settings, scripted_llm_factory):
        critic = CriticMasterAgent(scripted_llm_factory(["not json at all"]), settings)
        verdict = critic.review(make_doc(), [row()])
        assert verdict.source.startswith("rule")
        assert verdict.score == 10.0
        assert verdict.llm_error
