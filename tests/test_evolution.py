"""Evolution Log 的单测：写入、回读、few-shot 挑选、触发条件判定。"""

from __future__ import annotations

import json
from pathlib import Path

from ni101.config import AppConfig, LLMSpec, PipelineSettings
from ni101.evolution import (
    append_case,
    build_few_shots,
    load_cases,
    make_case,
    should_log,
    truncate,
)
from ni101.matcher import EvaluationResult
from ni101.pipeline import STATUS_ABSTAIN, STATUS_PARTIAL, STATUS_PASS, DocumentRun, RoundRecord
from ni101.schema import GroundTruthRow, ResourceRow


def make_run(status: str = STATUS_PASS, **kwargs) -> DocumentRun:
    run = DocumentRun(pdf_name="x.pdf", status=status)
    run.extraction_source = kwargs.pop("extraction_source", "llm")
    run.note = kwargs.pop("note", "passed")
    run.final_score = kwargs.pop("final_score", 10.0)
    run.rounds = kwargs.pop("rounds", [RoundRecord(round_index=1, score=10.0)])
    run.final_rows = kwargs.pop("final_rows", [])
    run.abstain = kwargs.pop("abstain", status == STATUS_ABSTAIN)
    run.abstain_reasons = kwargs.pop("abstain_reasons", [])
    return run


class TestAppendAndLoad:
    def test_round_trip(self, tmp_path: Path):
        path = tmp_path / "evolution.jsonl"
        append_case(path, {"case_id": "a", "reason": "第一条"})
        append_case(path, {"case_id": "b", "reason": "第二条"})
        cases = load_cases(path)
        assert [c["case_id"] for c in cases] == ["a", "b"]

    def test_creates_parent_directory(self, tmp_path: Path):
        path = tmp_path / "nested" / "deep" / "evolution.jsonl"
        append_case(path, {"case_id": "a"})
        assert path.exists()

    def test_missing_file_returns_empty(self, tmp_path: Path):
        assert load_cases(tmp_path / "absent.jsonl") == []

    def test_corrupt_line_is_skipped(self, tmp_path: Path):
        path = tmp_path / "evolution.jsonl"
        path.write_text(
            '{"case_id": "a"}\n{这不是 JSON\n{"case_id": "b"}\n', encoding="utf-8"
        )
        assert [c["case_id"] for c in load_cases(path)] == ["a", "b"]

    def test_truncate(self, tmp_path: Path):
        path = tmp_path / "evolution.jsonl"
        append_case(path, {"case_id": "a"})
        truncate(path)
        assert load_cases(path) == []


class TestFewShots:
    def test_only_usable_cases_selected(self):
        cases = [
            {"pdf": "a.pdf", "created_at": "2026-01-01", "predicted": [{"x": 1}],
             "corrected": [{"x": 2}], "reason": "可用"},
            {"pdf": "b.pdf", "created_at": "2026-01-02", "predicted": [],
             "corrected": [], "reason": "只有降级标记，没有对照"},
        ]
        picked = build_few_shots(cases, limit=5)
        assert len(picked) == 1
        assert picked[0]["reason"] == "可用"

    def test_most_recent_first_and_limited(self):
        cases = [
            {"pdf": f"{i}.pdf", "created_at": f"2026-01-0{i}", "predicted": [{"i": i}],
             "corrected": [{"i": i}]}
            for i in range(1, 6)
        ]
        picked = build_few_shots(cases, limit=2)
        assert [c["pdf"] for c in picked] == ["5.pdf", "4.pdf"]

    def test_excludes_same_pdf(self):
        cases = [
            {"pdf": "same.pdf", "created_at": "2026-01-02", "predicted": [{"i": 1}],
             "corrected": [{"i": 1}]},
            {"pdf": "other.pdf", "created_at": "2026-01-01", "predicted": [{"i": 2}],
             "corrected": [{"i": 2}]},
        ]
        picked = build_few_shots(cases, limit=5, exclude_pdf="same.pdf")
        assert [c["pdf"] for c in picked] == ["other.pdf"]

    def test_limit_zero(self):
        cases = [{"pdf": "a.pdf", "predicted": [1], "corrected": [1]}]
        assert build_few_shots(cases, limit=0) == []


class TestShouldLog:
    def test_clean_pass_is_not_logged(self):
        run = make_run(STATUS_PASS)
        ev = EvaluationResult(field_total=3, field_hit=3, accuracy_denominator_rows=1)
        config = AppConfig()
        logged, triggers = should_log(run, ev, config)
        assert logged is False
        assert triggers == []

    def test_document_abstain_is_logged(self):
        run = make_run(STATUS_ABSTAIN, note="critic_hard_abstain")
        logged, triggers = should_log(run, None, AppConfig())
        assert logged is True
        assert "document_abstained" in triggers

    def test_hard_give_is_logged(self):
        run = make_run(STATUS_PARTIAL, note="passed")
        ev = EvaluationResult(field_total=3, field_hit=3, accuracy_denominator_rows=1)
        ev.hard_give = ["inferred/li2o"]
        logged, triggers = should_log(run, ev, AppConfig())
        assert logged is True
        assert "hard_give" in triggers

    def test_wrong_but_confident_is_logged(self):
        run = make_run(STATUS_PASS, note="passed")
        ev = EvaluationResult(field_total=3, field_hit=1, accuracy_denominator_rows=1)
        logged, triggers = should_log(run, ev, AppConfig())
        assert logged is True
        assert "wrong_but_confident" in triggers

    def test_rule_fallback_is_logged(self):
        run = make_run(STATUS_PASS, extraction_source="rule")
        logged, triggers = should_log(
            run, EvaluationResult(field_total=0, field_hit=0), AppConfig()
        )
        assert logged is True
        assert "extraction_fallback" in triggers

    def test_rounds_exhausted_is_logged(self):
        run = make_run(STATUS_ABSTAIN, note="exhausted_3_rounds")
        _logged, triggers = should_log(run, None, AppConfig())
        assert "rounds_exhausted" in triggers

    def test_triggers_are_ordered_consistently(self):
        run = make_run(STATUS_ABSTAIN, note="exhausted_3_rounds", extraction_source="rule")
        _logged, triggers = should_log(run, None, AppConfig())
        assert triggers == ["document_abstained", "extraction_fallback", "rounds_exhausted"]


class TestMakeCase:
    def test_builds_predicted_and_corrected_from_field_errors(self):
        run = make_run(STATUS_PASS)
        run.final_rows = [
            ResourceRow(resource_class="indicated", commodity="Au", tonnage_mt=45.8,
                        grade=1.21, grade_unit="g/t", contained_metal=1782000,
                        contained_metal_unit="oz")
        ]
        ev = EvaluationResult(field_total=3, field_hit=2, accuracy_denominator_rows=1)
        from ni101.matcher import FieldComparison

        ev.field_comparisons = [
            FieldComparison("indicated/au", "tonnage_mt", 45.8, 45.8, True),
            FieldComparison("indicated/au", "grade", 1.21, 1.21, True),
            FieldComparison("indicated/au", "contained_metal", 1782000.0, 1782.0, False,
                            "超出 ±5% 容差"),
        ]
        truths = [
            GroundTruthRow(resource_class="indicated", commodity="Au", tonnage_mt=45.8,
                           grade=1.21, grade_unit="g/t", contained_metal=1782,
                           contained_metal_unit="koz")
        ]
        case = make_case(run, ev, truth_rows=truths, triggers=["wrong_but_confident"])

        assert case["pdf"] == "x.pdf"
        assert case["predicted"] and case["corrected"]
        assert "contained_metal" in json.dumps(case["predicted"])
        assert "1782" in json.dumps(case["corrected"])
        assert case["case_id"]

    def test_case_is_serializable(self):
        run = make_run(STATUS_ABSTAIN, abstain_reasons=["表格残缺"])
        case = make_case(run, None, triggers=["document_abstained"])
        json.dumps(case, ensure_ascii=False)
        assert "文档级拒答" in case["reason"]

    def test_few_shot_round_trip(self):
        run = make_run(STATUS_PASS)
        run.final_rows = [
            ResourceRow(resource_class="inferred", commodity="Au", tonnage_mt=23.6,
                        grade=0.94, grade_unit="g/t", contained_metal=713000,
                        contained_metal_unit="oz")
        ]
        ev = EvaluationResult(field_total=1, field_hit=0, accuracy_denominator_rows=1)
        from ni101.matcher import FieldComparison

        ev.field_comparisons = [
            FieldComparison("inferred/au", "contained_metal", 713000.0, 713.0, False)
        ]
        truths = [
            GroundTruthRow(resource_class="inferred", commodity="Au", contained_metal=713,
                           contained_metal_unit="koz")
        ]
        case = make_case(run, ev, truth_rows=truths)
        picked = build_few_shots([case], limit=3)
        assert len(picked) == 1
        assert picked[0]["predicted"] and picked[0]["corrected"]
