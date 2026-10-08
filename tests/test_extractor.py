"""Extractor Agent 的单测：规则路径的表格解析 + LLM 路径的 JSON 解析。"""

from __future__ import annotations

from pathlib import Path

import pytest

from ni101.extractor import ExtractorAgent
from ni101.llm import UnavailableLLM
from ni101.pdf import PdfDocument, PdfTable, load_pdf
from ni101.schema import GradeUnit, MetalUnit, ResourceClass

SINGLE_COMMODITY = [
    ["Resource Category", "Tonnes (Mt)", "Grade (g/t Au)", "Contained Au (koz)"],
    ["Indicated", "45.8", "1.21", "1,782"],
    ["Inferred", "23.6", "0.94", "713"],
    ["Total", "69.4", "-", "2,495"],
]

DUAL_COMMODITY = [
    ["Resource Category", "Tonnes (Mt)", "Cu (%)", "Au (g/t)", "Contained Cu (t)", "Contained Au (koz)"],
    ["Indicated", "320.5", "0.42", "0.55", "1,346,100", "5,668"],
    ["Inferred", "145.2", "0.38", "0.48", "551,760", "2,241"],
]

PERCENT_SINGLE = [
    ["Resource Category", "Tonnes (Mt)", "Li2O (%)", "Contained Li2O (Mt)"],
    ["Indicated", "64.3", "1.18", "0.76"],
    ["Inferred", "32.7", "1.05", "0.13"],
]


def make_doc(rows: list[list[str]], page_text: str = "Indicated Inferred Mineral Resources") -> PdfDocument:
    table = PdfTable(page_number=1, index_on_page=0, rows=rows)
    table.score = table.score_against(page_text)
    doc = PdfDocument(path=Path("synthetic.pdf"), page_count=1, page_texts=[page_text])
    doc.tables = [table]
    return doc


@pytest.fixture()
def agent() -> ExtractorAgent:
    from ni101.config import LLMSpec

    return ExtractorAgent(UnavailableLLM(LLMSpec(label="extractor")), verbose=False)


class TestRulePath:
    def test_single_commodity_gold(self, agent: ExtractorAgent):
        result = agent.extract(make_doc(SINGLE_COMMODITY))
        assert result.source == "rule"
        assert not result.abstain
        rows = {row.identity_key(): row for row in result.rows}
        assert set(rows) == {("indicated", "au"), ("inferred", "au")}

        indicated = rows[("indicated", "au")]
        assert indicated.tonnage_mt == 45.8
        assert indicated.grade == 1.21
        assert indicated.grade_unit is GradeUnit.G_T
        assert indicated.contained_metal == 1782.0
        assert indicated.contained_metal_unit is MetalUnit.KOZ
        assert indicated.source_page == 1

    def test_total_row_is_skipped(self, agent: ExtractorAgent):
        result = agent.extract(make_doc(SINGLE_COMMODITY))
        snippets = " ".join(row.source_snippet or "" for row in result.rows)
        assert "Total" not in snippets

    def test_dual_commodity_table(self, agent: ExtractorAgent):
        result = agent.extract(make_doc(DUAL_COMMODITY))
        rows = {row.identity_key(): row for row in result.rows}
        assert set(rows) == {
            ("indicated", "cu"),
            ("indicated", "au"),
            ("inferred", "cu"),
            ("inferred", "au"),
        }
        assert rows[("indicated", "cu")].grade == 0.42
        assert rows[("indicated", "cu")].grade_unit is GradeUnit.PERCENT
        assert rows[("indicated", "cu")].contained_metal_unit is MetalUnit.T
        assert rows[("indicated", "au")].contained_metal_unit is MetalUnit.KOZ

    def test_percent_grade_and_mt_metal(self, agent: ExtractorAgent):
        result = agent.extract(make_doc(PERCENT_SINGLE))
        rows = {row.identity_key(): row for row in result.rows}
        assert rows[("indicated", "li2o")].grade_unit is GradeUnit.PERCENT
        assert rows[("indicated", "li2o")].contained_metal_unit is MetalUnit.MT
        assert rows[("inferred", "li2o")].contained_metal == 0.13

    def test_scanned_pdf_abstains(self, agent: ExtractorAgent, tmp_path: Path):
        doc = PdfDocument(path=tmp_path / "scan.pdf", page_count=3, page_texts=["", "", ""])
        result = agent.extract(doc)
        assert result.abstain and not result.rows

    def test_no_table_abstains_instead_of_guessing(self, agent: ExtractorAgent):
        page_text = (
            "This document contains narrative text only. " * 12
            + "There is no mineral resource table on this page."
        )
        doc = PdfDocument(path=Path("empty.pdf"), page_count=1, page_texts=[page_text])
        result = agent.extract(doc)
        assert result.abstain
        assert any("未能" in reason for reason in result.abstain_reasons)

    def test_rule_path_cannot_revise(self, agent: ExtractorAgent):
        assert agent.source == "rule"
        assert agent.can_revise is False


class TestRealDemoPdfs:
    def test_extracts_expected_row_counts(self, agent: ExtractorAgent, demo_pdf_dir: Path):
        expected = {
            "newmont_ni43-101.pdf": 2,
            "barrick_ni43-101.pdf": 4,
            "pilbara_minerals_ni43-101.pdf": 2,
        }
        for name, count in expected.items():
            doc = load_pdf(demo_pdf_dir / name)
            result = agent.extract(doc)
            assert len(result.rows) == count, name
            assert {row.resource_class for row in result.rows} == {
                ResourceClass.INDICATED,
                ResourceClass.INFERRED,
            }


class TestLlmPath:
    def test_parses_llm_json(self, scripted_llm_factory, demo_pdf_dir: Path):
        payload = {
            "abstain": False,
            "abstain_reasons": [],
            "rows": [
                {
                    "resource_class": "indicated",
                    "commodity": "Au",
                    "tonnage_mt": "45.8",
                    "grade": "1.21",
                    "grade_unit": "g/t",
                    "contained_metal": "1,782",
                    "contained_metal_unit": "koz",
                    "source_page": 1,
                    "source_snippet": "Indicated 45.8 1.21 1,782",
                }
            ],
        }
        llm = scripted_llm_factory([payload])
        agent = ExtractorAgent(llm)
        assert agent.can_revise is True

        result = agent.extract(load_pdf(demo_pdf_dir / "newmont_ni43-101.pdf"))
        assert result.source == "llm"
        assert len(result.rows) == 1
        assert result.rows[0].contained_metal == 1782.0

    def test_strips_code_fence(self, scripted_llm_factory, demo_pdf_dir: Path):
        text = '```json\n{"rows": [{"resource_class": "inferred", "commodity": "Au", "tonnage_mt": 23.6}]}\n```'
        agent = ExtractorAgent(scripted_llm_factory([text]))
        result = agent.extract(load_pdf(demo_pdf_dir / "newmont_ni43-101.pdf"))
        assert result.source == "llm"
        assert result.rows[0].identity_key() == ("inferred", "au")

    def test_drops_empty_noise_rows(self, scripted_llm_factory, demo_pdf_dir: Path):
        payload = {
            "rows": [
                {"resource_class": "indicated", "commodity": "Au", "tonnage_mt": 45.8},
                {"resource_class": None, "commodity": "", "tonnage_mt": None, "grade": None,
                 "contained_metal": None},
            ]
        }
        agent = ExtractorAgent(scripted_llm_factory([payload]))
        result = agent.extract(load_pdf(demo_pdf_dir / "newmont_ni43-101.pdf"))
        assert len(result.rows) == 1

    def test_llm_garbage_falls_back_to_rules(self, scripted_llm_factory, demo_pdf_dir: Path):
        agent = ExtractorAgent(scripted_llm_factory(["这不是 JSON"]))
        result = agent.extract(load_pdf(demo_pdf_dir / "newmont_ni43-101.pdf"))
        # 模型抽不出来时用规则兜底，但必须如实标注来源
        assert result.source == "rule"
        assert "llm_failed_fallback_rule" in result.note
        assert len(result.rows) == 2

    def test_feedback_and_few_shots_reach_prompt(self, scripted_llm_factory, demo_pdf_dir: Path):
        payload = {"rows": [{"resource_class": "indicated", "commodity": "Au", "tonnage_mt": 45.8}]}
        llm = scripted_llm_factory([payload])
        agent = ExtractorAgent(llm)
        agent.extract(
            load_pdf(demo_pdf_dir / "newmont_ni43-101.pdf"),
            feedback=[{"row_index": 0, "field": "grade", "severity": "major", "message": "品位缺失"}],
            few_shots=[
                {
                    "pdf": "old.pdf",
                    "reason": "历史上把 koz 当成了 oz",
                    "predicted": [{"commodity": "Au", "contained_metal": 1782000}],
                    "corrected": [{"commodity": "Au", "contained_metal": 1782}],
                }
            ],
        )
        prompt = llm.prompts[0]
        assert "上一轮审核发现的问题" in prompt
        assert "品位缺失" in prompt
        assert "历史失败案例" in prompt
