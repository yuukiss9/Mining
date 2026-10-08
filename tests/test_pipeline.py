"""Revise Loop 编排的单测：轮次上限、反馈回灌、拒答分支、规则路径不空转。"""

from __future__ import annotations

from pathlib import Path

import pytest

from ni101.config import LLMSpec
from ni101.critic import CriticMasterAgent
from ni101.extractor import ExtractorAgent
from ni101.llm import UnavailableLLM
from ni101.pdf import PdfDocument, PdfTable
from ni101.pipeline import (
    STATUS_ABSTAIN,
    STATUS_PARTIAL,
    STATUS_PASS,
    run_document,
)

PAGE_TEXT = (
    "Mineral Resource Statement. Indicated Mineral Resources are reported "
    "above a cut-off grade of 0.50 g/t Au."
)

WITH_UNITS = [
    ["Resource Category", "Tonnes (Mt)", "Grade (g/t Au)", "Contained Au (koz)"],
    ["Indicated", "45.8", "1.21", "1,782"],
    ["Inferred", "23.6", "0.94", "713"],
]

WITHOUT_UNITS = [
    ["Resource Category", "Tonnes", "Grade", "Contained"],
    ["Indicated", "45.8", "1.21", "1,782"],
    ["Inferred", "23.6", "0.94", "713"],
]

INCONSISTENT_LITHIUM = [
    ["Resource Category", "Tonnes (Mt)", "Li2O (%)", "Contained Li2O (Mt)"],
    ["Indicated", "64.3", "1.18", "0.76"],
    ["Inferred", "32.7", "1.05", "0.13"],
]

GOOD_ROWS = {
    "rows": [
        {
            "resource_class": "indicated",
            "commodity": "Au",
            "tonnage_mt": 45.8,
            "grade": 1.21,
            "grade_unit": "g/t",
            "contained_metal": 1782,
            "contained_metal_unit": "koz",
            "source_page": 1,
            "source_snippet": "Indicated 45.8 1.21 1,782",
        },
        {
            "resource_class": "inferred",
            "commodity": "Au",
            "tonnage_mt": 23.6,
            "grade": 0.94,
            "grade_unit": "g/t",
            "contained_metal": 713,
            "contained_metal_unit": "koz",
            "source_page": 1,
            "source_snippet": "Inferred 23.6 0.94 713",
        },
    ]
}

#: 只给了吨位，其余四个字段全缺 -> 4 x major = 6 分扣完，得分 4.0
INCOMPLETE_ROWS = {
    "rows": [
        {
            "resource_class": "indicated",
            "commodity": "Au",
            "tonnage_mt": 45.8,
            "source_page": 1,
            "source_snippet": "Indicated 45.8",
        }
    ]
}


def make_doc(rows: list[list[str]], page_text: str = PAGE_TEXT) -> PdfDocument:
    table = PdfTable(page_number=1, index_on_page=0, rows=rows)
    table.score = table.score_against(page_text)
    doc = PdfDocument(path=Path("synthetic.pdf"), page_count=1, page_texts=[page_text])
    doc.tables = [table]
    return doc


def make_agents(config, extractor_llm, critic_llm=None):
    return (
        ExtractorAgent(extractor_llm),
        CriticMasterAgent(critic_llm or UnavailableLLM(LLMSpec(label="critic")), config.pipeline),
    )


class TestReviseLoop:
    def test_passes_on_second_round_with_feedback(self, config, scripted_llm_factory):
        llm = scripted_llm_factory([INCOMPLETE_ROWS, GOOD_ROWS])
        extractor, critic = make_agents(config, llm)
        run = run_document(make_doc(WITH_UNITS), config, extractor, critic)

        assert run.status == STATUS_PASS
        assert len(run.rounds) == 2
        assert run.rounds[0].score < config.pipeline.pass_score
        assert run.rounds[1].score >= config.pipeline.pass_score
        assert run.final_score >= config.pipeline.pass_score
        # 第 2 轮的 prompt 必须带上第 1 轮 Critic 提出的问题
        assert "上一轮审核发现的问题" in llm.prompts[1]
        assert "contained_metal" in llm.prompts[1]

    def test_passes_on_first_round(self, config, scripted_llm_factory):
        extractor, critic = make_agents(config, scripted_llm_factory([GOOD_ROWS]))
        run = run_document(make_doc(WITH_UNITS), config, extractor, critic)
        assert run.status == STATUS_PASS
        assert len(run.rounds) == 1

    def test_exhausts_rounds_then_abstains(self, config, scripted_llm_factory):
        extractor, critic = make_agents(config, scripted_llm_factory([INCOMPLETE_ROWS]))
        run = run_document(make_doc(WITH_UNITS), config, extractor, critic)

        assert len(run.rounds) == config.pipeline.max_rounds == 3
        assert run.status == STATUS_ABSTAIN
        assert run.abstain is True
        assert run.note == "exhausted_3_rounds"
        assert run.abstain_reasons

    def test_respects_custom_max_rounds(self, config, scripted_llm_factory):
        config.pipeline.max_rounds = 2
        extractor, critic = make_agents(config, scripted_llm_factory([INCOMPLETE_ROWS]))
        run = run_document(make_doc(WITH_UNITS), config, extractor, critic)
        assert len(run.rounds) == 2
        assert run.note == "exhausted_2_rounds"

    def test_few_shots_reach_extractor(self, config, scripted_llm_factory):
        llm = scripted_llm_factory([GOOD_ROWS])
        extractor, critic = make_agents(config, llm)
        run_document(
            make_doc(WITH_UNITS),
            config,
            extractor,
            critic,
            few_shots=[
                {
                    "pdf": "old.pdf",
                    "reason": "把 koz 当成了 oz",
                    "predicted": [{"commodity": "Au", "contained_metal": 1782000}],
                    "corrected": [{"commodity": "Au", "contained_metal": 1782}],
                }
            ],
        )
        assert "历史失败案例" in llm.prompts[0]


class TestRulePathDoesNotSpin:
    def test_no_revision_capability_abstains_immediately(self, config):
        """规则路径是确定性的，喂回反馈也不会变，不该空跑 3 轮。"""
        extractor, critic = make_agents(config, UnavailableLLM(LLMSpec(label="extractor")))
        run = run_document(make_doc(WITHOUT_UNITS), config, extractor, critic)

        assert len(run.rounds) == 1
        assert run.status == STATUS_ABSTAIN
        assert run.note == "no_revision_capability"
        assert run.final_score < config.pipeline.pass_score

    def test_clean_rule_extraction_passes(self, config):
        extractor, critic = make_agents(config, UnavailableLLM(LLMSpec(label="extractor")))
        run = run_document(make_doc(WITH_UNITS), config, extractor, critic)
        assert run.status == STATUS_PASS
        assert len(run.rounds) == 1
        assert run.extraction_source == "rule"


class TestRowLevelAbstain:
    def test_contradictory_row_abstains_others_pass(self, config):
        extractor, critic = make_agents(config, UnavailableLLM(LLMSpec(label="extractor")))
        run = run_document(make_doc(INCONSISTENT_LITHIUM), config, extractor, critic)

        assert run.status == STATUS_PARTIAL
        assert run.abstain is False
        abstained = [row for row in run.final_rows if row.abstain]
        assert len(abstained) == 1
        assert abstained[0].identity_key() == ("inferred", "li2o")
        assert abstained[0].abstain_reason
        # 干净的那一行仍保留
        kept = [row for row in run.final_rows if not row.abstain]
        assert kept[0].identity_key() == ("indicated", "li2o")
        assert run.abstain_reasons


class TestDegradedBranches:
    def test_scanned_pdf_abstains_without_ocr(self, config):
        extractor, critic = make_agents(config, UnavailableLLM(LLMSpec(label="extractor")))
        doc = PdfDocument(path=Path("scan.pdf"), page_count=4, page_texts=[""] * 4)
        run = run_document(doc.path, config, extractor, critic, document=doc)
        assert run.status == STATUS_ABSTAIN
        assert run.note == "scanned_pdf"
        assert any("OCR" in reason for reason in run.abstain_reasons)

    def test_missing_pdf_is_handled(self, config, tmp_path):
        extractor, critic = make_agents(config, UnavailableLLM(LLMSpec(label="extractor")))
        run = run_document(tmp_path / "nope.pdf", config, extractor, critic)
        assert run.status == STATUS_ABSTAIN
        assert run.note == "pdf_read_failed"

    def test_extractor_error_is_recorded(self, config, scripted_llm_factory):
        extractor, critic = make_agents(config, scripted_llm_factory(["not json"]))
        run = run_document(make_doc(WITH_UNITS), config, extractor, critic)
        assert run.rounds[0].error
        assert run.extraction_source == "llm"

    def test_run_serialization(self, config):
        extractor, critic = make_agents(config, UnavailableLLM(LLMSpec(label="extractor")))
        run = run_document(make_doc(WITH_UNITS), config, extractor, critic)
        payload = run.to_dict()
        assert payload["pdf"] == "synthetic.pdf"
        assert payload["rounds"][0]["score"] == pytest.approx(10.0)
        assert payload["final_rows"]
