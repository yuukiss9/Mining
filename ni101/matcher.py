"""字段级比对与评分协议（容差 ±5%）。

防"假失败"是本模块的核心目标——机械比对极易把**正确的输出**判成错，
一旦判错就会反复重试直到转人工，主流程永远走不到成功。三类假失败在这题里都真实存在：

| 假失败形态 | 例子 | 本模块的修法 |
| --- | --- | --- |
| 单位 | 原文 `1,782 koz`，模型给 `1,782,000 oz` | 统一归一到公吨 / g/t 后再比 |
| 精度 | 原文 `0.76 Mt`，模型给 `758,740 t` | 相对容差 ±5%，不提前 round |
| 写法 | 原文 `Li2O`，模型给 `Li` | 先行精确匹配，再按资源类别兜底配对 |

**分母口径**：只有 ground truth 中 ``expect_abstain=False`` 的行才进入准确率分母。
标了 ``expect_abstain=True`` 的行（原文自相矛盾）走拒答通道单独统计，
不能混进准确率——否则噪声会盖住真问题。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .schema import (
    GroundTruthDocument,
    GroundTruthRow,
    ResourceRow,
    to_gr_per_tonne,
    to_tonnes,
)

#: 参与比对的三个数值字段，以及把预测值归一化到 GT 口径的函数
_NUMERIC_FIELDS: tuple[str, ...] = ("tonnage_mt", "grade", "contained_metal")


def load_ground_truth(path: Path) -> GroundTruthDocument:
    """读取 ground truth JSON。

    **题目给的 GT 结构若与本项目不同，只改这一个函数即可**，其余代码不动。
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        raw = {"rows": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} 顶层应为对象或数组")
    raw.setdefault("pdf", path.stem + ".pdf")
    return GroundTruthDocument(**raw)


def find_ground_truth(pdf_path: Path, gt_dir: Path) -> Optional[Path]:
    """按 PDF 的 stem 在 GT 目录里找同名 JSON（后缀大小写不敏感）。"""
    stem = Path(pdf_path).stem
    if not gt_dir.exists():
        return None
    for candidate in gt_dir.glob("*.json"):
        if candidate.stem.lower() == stem.lower():
            return candidate
    return None


# --------------------------------------------------------------------------
# 归一化
# --------------------------------------------------------------------------
def _normalized(row: ResourceRow | GroundTruthRow) -> dict[str, Optional[float]]:
    """把三个数值字段归一到同一口径：吨位 Mt、品位 g/t、金属量 t。"""
    return {
        "tonnage_mt": row.tonnage_mt,
        "grade": to_gr_per_tonne(row.grade, row.grade_unit),
        "contained_metal": to_tonnes(row.contained_metal, row.contained_metal_unit),
    }


def _close(predicted: Optional[float], expected: Optional[float], tol: float) -> bool:
    if predicted is None or expected is None:
        return False
    scale = max(abs(expected), 1e-12)
    return abs(predicted - expected) <= tol * scale


def _key(row: ResourceRow | GroundTruthRow) -> tuple[str, str]:
    return row.identity_key()


# --------------------------------------------------------------------------
# 结果容器
# --------------------------------------------------------------------------
@dataclass
class FieldComparison:
    row_key: str
    field: str
    predicted: Optional[float]
    expected: Optional[float]
    matched: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row_key,
            "field": self.field,
            "predicted": self.predicted,
            "expected": self.expected,
            "matched": self.matched,
            "reason": self.reason,
        }


@dataclass
class EvaluationResult:
    pdf: str = ""
    gt_rows: int = 0
    pred_rows: int = 0
    matched_rows: int = 0
    field_total: int = 0
    field_hit: int = 0
    accuracy_denominator_rows: int = 0

    field_comparisons: list[FieldComparison] = field(default_factory=list)
    correct_abstain: list[str] = field(default_factory=list)
    hard_give: list[str] = field(default_factory=list)
    missing_rows: list[str] = field(default_factory=list)
    extra_rows: list[str] = field(default_factory=list)
    abstained_rows: list[str] = field(default_factory=list)
    unmatched_predictions: list[str] = field(default_factory=list)

    @property
    def field_accuracy(self) -> float:
        if not self.field_total:
            return 0.0
        return self.field_hit / self.field_total

    @property
    def row_precision(self) -> float:
        if not self.pred_rows:
            return 0.0
        return self.matched_rows / self.pred_rows

    @property
    def row_recall(self) -> float:
        expected = self.gt_rows - len(self.correct_abstain) - len(self.hard_give)
        if expected <= 0:
            # 全部是"应当拒答"的行时，召回率不适用，返回 1.0 避免污染均值
            return 1.0
        return self.matched_rows / expected

    @property
    def expected_abstain_count(self) -> int:
        return len(self.correct_abstain) + len(self.hard_give)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pdf": self.pdf,
            "gt_rows": self.gt_rows,
            "pred_rows": self.pred_rows,
            "matched_rows": self.matched_rows,
            "field_total": self.field_total,
            "field_hit": self.field_hit,
            "field_accuracy": round(self.field_accuracy, 4),
            "row_precision": round(self.row_precision, 4),
            "row_recall": round(self.row_recall, 4),
            "correct_abstain": self.correct_abstain,
            "hard_give": self.hard_give,
            "abstained_rows": self.abstained_rows,
            "missing_rows": self.missing_rows,
            "extra_rows": self.extra_rows,
            "field_comparisons": [c.to_dict() for c in self.field_comparisons],
        }


# --------------------------------------------------------------------------
# 主比对
# --------------------------------------------------------------------------
def _pair_rows(
    predictions: Sequence[ResourceRow], truths: Sequence[GroundTruthRow]
) -> tuple[dict[int, int], list[int], list[int]]:
    """预测行与 GT 行配对：先按 (类别, 矿种) 精确匹配，再按资源类别兜底。"""
    pairs: dict[int, int] = {}          # gt_index -> pred_index
    used_pred: set[int] = set()
    pred_keys = {i: _key(row) for i, row in enumerate(predictions)}

    for gt_index, truth in enumerate(truths):
        target = _key(truth)
        for pred_index, key in pred_keys.items():
            if pred_index in used_pred or key != target:
                continue
            pairs[gt_index] = pred_index
            used_pred.add(pred_index)
            break

    # 兜底：同资源类别的未匹配行按顺序配对（应对矿种写法差异）
    for gt_index, truth in enumerate(truths):
        if gt_index in pairs:
            continue
        for pred_index, row in enumerate(predictions):
            if pred_index in used_pred:
                continue
            if row.resource_class is not None and row.resource_class == truth.resource_class:
                pairs[gt_index] = pred_index
                used_pred.add(pred_index)
                break

    unmatched_pred = [i for i in range(len(predictions)) if i not in used_pred]
    unmatched_gt = [i for i in range(len(truths)) if i not in pairs]
    return pairs, unmatched_gt, unmatched_pred


def evaluate(
    pdf_name: str,
    predictions: Sequence[ResourceRow],
    truth_doc: Optional[GroundTruthDocument],
    tolerance: float = 0.05,
    document_abstained: bool = False,
) -> EvaluationResult:
    """按评分协议（字段级 accuracy，容差 ±5%）比对。

    ``document_abstained=True`` 表示系统对整篇文档拒答：此时所有 GT 行都记为
    「系统整体拒答」，不计入准确率分母，也不记 hard_give（拒答比硬给安全）。
    """
    result = EvaluationResult(pdf=pdf_name, pred_rows=len(predictions))
    if truth_doc is None:
        result.unmatched_predictions = [row.summary() for row in predictions]
        return result

    # 行精确率的分母只算"真的给出了答案"的行：拒答的行是主动放弃，
    # 不该因为"拒答了这个类别"而被算成错误预测。
    result.pred_rows = sum(1 for row in predictions if not row.abstain)

    truths = truth_doc.rows
    result.gt_rows = len(truths)

    if document_abstained and not predictions:
        for truth in truths:
            label = f"{truth.identity_key()[0]}/{truth.identity_key()[1]}"
            result.abstained_rows.append(label)
        return result

    pairs, unmatched_gt, unmatched_pred = _pair_rows(predictions, truths)

    for gt_index, truth in enumerate(truths):
        label = f"{truth.identity_key()[0]}/{truth.identity_key()[1]}"
        pred_index = pairs.get(gt_index)

        if truth.expect_abstain:
            if pred_index is None:
                # 没抽到这一行：既没硬给也没明确拒答，记为缺失（比硬给轻）
                result.missing_rows.append(label)
                result.correct_abstain.append(label + "（未输出，等同未硬给）")
            elif predictions[pred_index].abstain:
                result.correct_abstain.append(label)
            else:
                result.hard_give.append(label)
            continue

        if pred_index is None:
            result.missing_rows.append(label)
            continue

        prediction = predictions[pred_index]
        if prediction.abstain:
            # GT 认为该行可抽，但系统拒答 —— 过度保守，记入 abstained_rows
            result.abstained_rows.append(label)
            result.missing_rows.append(label)
            continue

        result.matched_rows += 1
        result.accuracy_denominator_rows += 1
        gt_values = _normalized(truth)
        pred_values = _normalized(prediction)

        for field_name in _NUMERIC_FIELDS:
            expected = gt_values[field_name]
            predicted = pred_values[field_name]
            result.field_total += 1
            if expected is None:
                # GT 本身没这个值，不计入正确也不计入错误（分母已包含）
                matched = predicted is None
                reason = "GT 侧为空" if matched else "GT 侧为空但预测给了值"
            elif predicted is None:
                matched = False
                reason = "缺失"
            else:
                matched = _close(predicted, expected, tolerance)
                reason = "" if matched else f"超出 ±{tolerance:.0%} 容差"
            if matched:
                result.field_hit += 1
            result.field_comparisons.append(
                FieldComparison(
                    row_key=label,
                    field=field_name,
                    predicted=predicted,
                    expected=expected,
                    matched=matched,
                    reason=reason,
                )
            )

    for pred_index in unmatched_pred:
        row = predictions[pred_index]
        if row.abstain:
            result.abstained_rows.append(f"{row.identity_key()}（无对应 GT）")
        else:
            result.extra_rows.append(row.summary())

    return result


def evaluate_run(
    run: Any, truth_doc: Optional[GroundTruthDocument], tolerance: float
) -> EvaluationResult:
    """:class:`~ni101.pipeline.DocumentRun` 的便捷包装。"""
    return evaluate(
        run.pdf_name,
        run.final_rows,
        truth_doc,
        tolerance=tolerance,
        document_abstained=bool(getattr(run, "abstain", False)),
    )


def as_eval_dict(result: EvaluationResult) -> dict[str, Any]:
    return result.to_dict()


def collect_field_errors(result: EvaluationResult) -> Iterable[FieldComparison]:
    return (c for c in result.field_comparisons if not c.matched)
