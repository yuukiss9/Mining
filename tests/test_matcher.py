"""评分协议（字段级 accuracy，容差 ±5%）与防"假失败"的单测。"""

from __future__ import annotations

import pytest

from ni101.matcher import _close, evaluate, load_ground_truth
from ni101.pipeline import DocumentRun
from ni101.schema import GroundTruthDocument, GroundTruthRow, ResourceRow

TOL = 0.05


def pred(resource_class: str, commodity: str, tonnage, grade, grade_unit, metal, metal_unit,
         abstain: bool = False) -> ResourceRow:
    return ResourceRow(
        resource_class=resource_class,
        commodity=commodity,
        tonnage_mt=tonnage,
        grade=grade,
        grade_unit=grade_unit,
        contained_metal=metal,
        contained_metal_unit=metal_unit,
        source_page=1,
        source_snippet="x",
        abstain=abstain,
    )


def truth(resource_class: str, commodity: str, tonnage, grade, grade_unit, metal, metal_unit,
          expect_abstain: bool = False) -> GroundTruthRow:
    return GroundTruthRow(
        resource_class=resource_class,
        commodity=commodity,
        tonnage_mt=tonnage,
        grade=grade,
        grade_unit=grade_unit,
        contained_metal=metal,
        contained_metal_unit=metal_unit,
        source_page=1,
        expect_abstain=expect_abstain,
    )


class TestTolerance:
    def test_exact_match(self):
        assert _close(45.8, 45.8, TOL)

    def test_within_5_percent(self):
        assert _close(46.0, 45.8, TOL)          # +0.44%
        assert _close(47.5, 45.8, TOL)          # +3.7%

    def test_outside_5_percent(self):
        assert not _close(48.5, 45.8, TOL)      # +5.9%

    def test_none_is_never_close(self):
        assert not _close(None, 45.8, TOL)
        assert not _close(45.8, None, TOL)


class TestUnitInvariance:
    """同一物理量换单位表达，必须判为"对"——否则会一直重试到转人工。"""

    def test_koz_vs_oz(self):
        truth_doc = GroundTruthDocument(
            rows=[truth("indicated", "Au", 45.8, 1.21, "g/t", 1782, "koz")]
        )
        predictions = [pred("indicated", "Au", 45.8, 1.21, "g/t", 1_782_000, "oz")]
        result = evaluate("x.pdf", predictions, truth_doc, tolerance=TOL)
        assert result.field_hit == result.field_total == 3

    def test_mt_vs_t(self):
        truth_doc = GroundTruthDocument(
            rows=[truth("indicated", "Li2O", 64.3, 1.18, "%", 0.76, "Mt")]
        )
        predictions = [pred("indicated", "Li2O", 64.3, 1.18, "%", 760_000, "t")]
        result = evaluate("x.pdf", predictions, truth_doc, tolerance=TOL)
        assert result.field_accuracy == 1.0

    def test_grade_percent_vs_g_t(self):
        truth_doc = GroundTruthDocument(
            rows=[truth("indicated", "Cu", 320.5, 0.42, "%", 1_346_100, "t")]
        )
        predictions = [pred("indicated", "Cu", 320.5, 4200, "g/t", 1_346_100, "t")]
        result = evaluate("x.pdf", predictions, truth_doc, tolerance=TOL)
        assert result.field_accuracy == 1.0


class TestRowPairing:
    def test_matches_by_class_and_commodity(self):
        truth_doc = GroundTruthDocument(
            rows=[
                truth("indicated", "Cu", 320.5, 0.42, "%", 1_346_100, "t"),
                truth("indicated", "Au", 320.5, 0.55, "g/t", 5668, "koz"),
            ]
        )
        predictions = [
            pred("indicated", "Au", 320.5, 0.55, "g/t", 5668, "koz"),
            pred("indicated", "Cu", 320.5, 0.42, "%", 1_346_100, "t"),
        ]
        result = evaluate("x.pdf", predictions, truth_doc, tolerance=TOL)
        assert result.matched_rows == 2
        assert result.field_accuracy == 1.0
        assert not result.extra_rows

    def test_falls_back_to_class_when_commodity_differs(self):
        truth_doc = GroundTruthDocument(
            rows=[truth("indicated", "Li2O", 64.3, 1.18, "%", 0.76, "Mt")]
        )
        predictions = [pred("indicated", "Li", 64.3, 1.18, "%", 0.76, "Mt")]
        result = evaluate("x.pdf", predictions, truth_doc, tolerance=TOL)
        assert result.matched_rows == 1

    def test_missing_and_extra_rows_reported(self):
        truth_doc = GroundTruthDocument(
            rows=[
                truth("indicated", "Au", 45.8, 1.21, "g/t", 1782, "koz"),
                truth("inferred", "Au", 23.6, 0.94, "g/t", 713, "koz"),
            ]
        )
        predictions = [pred("indicated", "Au", 45.8, 1.21, "g/t", 1782, "koz")]
        result = evaluate("x.pdf", predictions, truth_doc, tolerance=TOL)
        assert result.matched_rows == 1
        assert result.missing_rows == ["inferred/au"]


class TestAbstainProtocol:
    """题目最看重的一条：明显错误时应 abstain 而不是硬给。"""

    def test_correct_abstain(self):
        truth_doc = GroundTruthDocument(
            rows=[truth("inferred", "Li2O", 32.7, 1.05, "%", 0.13, "Mt", expect_abstain=True)]
        )
        predictions = [pred("inferred", "Li2O", 32.7, 1.05, "%", 0.13, "Mt", abstain=True)]
        result = evaluate("x.pdf", predictions, truth_doc, tolerance=TOL)
        assert result.correct_abstain == ["inferred/li2o"]
        assert not result.hard_give
        # 拒答的行不进入准确率分母
        assert result.field_total == 0
        # 拒答的行也不该拉低行精确率
        assert result.pred_rows == 0

    def test_hard_give_is_flagged(self):
        truth_doc = GroundTruthDocument(
            rows=[truth("inferred", "Li2O", 32.7, 1.05, "%", 0.13, "Mt", expect_abstain=True)]
        )
        predictions = [pred("inferred", "Li2O", 32.7, 1.05, "%", 0.13, "Mt", abstain=False)]
        result = evaluate("x.pdf", predictions, truth_doc, tolerance=TOL)
        assert result.hard_give == ["inferred/li2o"]
        assert not result.correct_abstain

    def test_document_level_abstain(self):
        truth_doc = GroundTruthDocument(
            rows=[
                truth("indicated", "Au", 45.8, 1.21, "g/t", 1782, "koz"),
                truth("inferred", "Au", 23.6, 0.94, "g/t", 713, "koz"),
            ]
        )
        result = evaluate("x.pdf", [], truth_doc, tolerance=TOL, document_abstained=True)
        assert result.abstained_rows == ["indicated/au", "inferred/au"]
        assert result.field_total == 0
        assert not result.hard_give

    def test_over_abstain_on_solvable_row(self):
        truth_doc = GroundTruthDocument(
            rows=[truth("indicated", "Au", 45.8, 1.21, "g/t", 1782, "koz")]
        )
        predictions = [pred("indicated", "Au", None, None, None, None, None, abstain=True)]
        result = evaluate("x.pdf", predictions, truth_doc, tolerance=TOL)
        assert result.abstained_rows == ["indicated/au"]
        assert result.matched_rows == 0


class TestMetricProperties:
    def test_precision_recall(self):
        truth_doc = GroundTruthDocument(
            rows=[
                truth("indicated", "Au", 45.8, 1.21, "g/t", 1782, "koz"),
                truth("inferred", "Au", 23.6, 0.94, "g/t", 713, "koz"),
            ]
        )
        predictions = [
            pred("indicated", "Au", 45.8, 1.21, "g/t", 1782, "koz"),
            pred("indicated", "Ag", 99.0, 1.0, "g/t", 1.0, "koz"),
        ]
        result = evaluate("x.pdf", predictions, truth_doc, tolerance=TOL)
        assert result.pred_rows == 2
        assert result.matched_rows == 1
        assert result.row_precision == 0.5
        assert result.row_recall == 0.5

    def test_ground_truth_loader_accepts_list(self, tmp_path):
        path = tmp_path / "gt.json"
        path.write_text(
            '[{"resource_class": "indicated", "commodity": "Au", "tonnage_mt": 1}]',
            encoding="utf-8",
        )
        doc = load_ground_truth(path)
        assert doc.pdf == "gt.pdf"
        assert doc.rows[0].commodity == "Au"


class TestEvaluateRun:
    def test_wrapper_reads_run_status(self):
        run = DocumentRun(pdf_name="x.pdf", abstain=True)
        truth_doc = GroundTruthDocument(rows=[truth("indicated", "Au", 1, 1, "g/t", 1, "koz")])
        result = evaluate("x.pdf", run.final_rows, truth_doc, tolerance=TOL,
                          document_abstained=run.abstain)
        assert result.abstained_rows == ["indicated/au"]
