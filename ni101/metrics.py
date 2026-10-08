"""指标汇总与报告渲染。

核心指标是题目明确说的那一句：**当抽取明显错误时，系统是否 abstain 而不是硬给**。
对应两个数：

* ``hard_give_rate``（硬给率，越低越好）—— GT 标记「应当拒答」的行里，被硬给了数值的比例。
* ``wrong_but_confident_docs``（硬给文档）—— 未拒答但字段准确率不达标的文档数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .config import AppConfig
from .matcher import EvaluationResult
from .pipeline import STATUS_ABSTAIN, STATUS_PARTIAL, STATUS_PASS, DocumentRun


@dataclass
class DocSummary:
    pdf: str
    status: str
    field_accuracy: float
    row_precision: float
    row_recall: float
    rounds: int
    final_score: float
    extraction_source: str
    hard_give: int
    correct_abstain: int
    is_wrong_but_confident: bool


@dataclass
class SuiteSummary:
    docs: int = 0
    status_counts: dict[str, int] = None  # type: ignore[assignment]
    field_total: int = 0
    field_hit: int = 0
    matched_rows: int = 0
    pred_rows: int = 0
    gt_rows: int = 0
    expected_abstain_total: int = 0
    correct_abstain_total: int = 0
    hard_give_total: int = 0
    wrong_but_confident_docs: int = 0
    evaluations: Sequence[EvaluationResult] = ()
    per_doc: list[DocSummary] = None  # type: ignore[assignment]

    @property
    def field_accuracy(self) -> float:
        return self.field_hit / self.field_total if self.field_total else 0.0

    @property
    def row_precision(self) -> float:
        return self.matched_rows / self.pred_rows if self.pred_rows else 0.0

    @property
    def row_recall(self) -> float:
        denominator = self.gt_rows - self.correct_abstain_total - self.hard_give_total
        if denominator <= 0:
            return 1.0
        return self.matched_rows / denominator

    @property
    def hard_give_rate(self) -> float:
        if not self.expected_abstain_total:
            return 0.0
        return self.hard_give_total / self.expected_abstain_total

    @property
    def correct_abstain_rate(self) -> float:
        if not self.expected_abstain_total:
            return 0.0
        return self.correct_abstain_total / self.expected_abstain_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "docs": self.docs,
            "status_counts": self.status_counts or {},
            "field_total": self.field_total,
            "field_hit": self.field_hit,
            "field_accuracy": round(self.field_accuracy, 4),
            "row_precision": round(self.row_precision, 4),
            "row_recall": round(self.row_recall, 4),
            "expected_abstain_total": self.expected_abstain_total,
            "correct_abstain_total": self.correct_abstain_total,
            "hard_give_total": self.hard_give_total,
            "hard_give_rate": round(self.hard_give_rate, 4),
            "correct_abstain_rate": round(self.correct_abstain_rate, 4),
            "wrong_but_confident_docs": self.wrong_but_confident_docs,
            "per_doc": [vars(doc) for doc in (self.per_doc or [])],
        }


def summarize(
    pairs: Sequence[tuple[DocumentRun, Optional[EvaluationResult]]],
    config: AppConfig,
) -> SuiteSummary:
    summary = SuiteSummary(
        docs=len(pairs),
        status_counts={STATUS_PASS: 0, STATUS_PARTIAL: 0, STATUS_ABSTAIN: 0},
        per_doc=[],
        evaluations=tuple(ev for _, ev in pairs if ev is not None),
    )
    threshold = config.pipeline.min_field_accuracy_to_pass

    for run, evaluation in pairs:
        summary.status_counts[run.status] = summary.status_counts.get(run.status, 0) + 1

        if evaluation is None:
            summary.per_doc.append(
                DocSummary(
                    pdf=run.pdf_name,
                    status=run.status,
                    field_accuracy=0.0,
                    row_precision=0.0,
                    row_recall=0.0,
                    rounds=len(run.rounds),
                    final_score=run.final_score,
                    extraction_source=run.extraction_source,
                    hard_give=0,
                    correct_abstain=0,
                    is_wrong_but_confident=False,
                )
            )
            continue

        summary.field_total += evaluation.field_total
        summary.field_hit += evaluation.field_hit
        summary.matched_rows += evaluation.matched_rows
        summary.pred_rows += evaluation.pred_rows
        summary.gt_rows += evaluation.gt_rows
        summary.expected_abstain_total += evaluation.expected_abstain_count
        summary.correct_abstain_total += len(evaluation.correct_abstain)
        summary.hard_give_total += len(evaluation.hard_give)

        wrong_but_confident = (
            run.status != STATUS_ABSTAIN
            and evaluation.accuracy_denominator_rows > 0
            and evaluation.field_accuracy < threshold
        )
        if wrong_but_confident:
            summary.wrong_but_confident_docs += 1

        summary.per_doc.append(
            DocSummary(
                pdf=run.pdf_name,
                status=run.status,
                field_accuracy=evaluation.field_accuracy,
                row_precision=evaluation.row_precision,
                row_recall=evaluation.row_recall,
                rounds=len(run.rounds),
                final_score=run.final_score,
                extraction_source=run.extraction_source,
                hard_give=len(evaluation.hard_give),
                correct_abstain=len(evaluation.correct_abstain),
                is_wrong_but_confident=wrong_but_confident,
            )
        )

    return summary


def render_markdown_report(
    summary: SuiteSummary,
    pairs: Sequence[tuple[DocumentRun, Optional[EvaluationResult]]],
    config: AppConfig,
) -> str:
    """渲染一份可直接粘贴进汇报的 markdown 报告。"""
    lines: list[str] = []
    lines.append("# NI 43-101 资源量抽取 —— 对抗审核评测报告")
    lines.append("")
    lines.append("## 1. 总览指标")
    lines.append("")
    lines.append("| 指标 | 数值 | 说明 |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| 文档数 | {summary.docs} | |")
    lines.append(
        f"| 状态分布 | 通过 {summary.status_counts.get(STATUS_PASS, 0)} / "
        f"部分通过 {summary.status_counts.get(STATUS_PARTIAL, 0)} / "
        f"拒答 {summary.status_counts.get(STATUS_ABSTAIN, 0)} | |"
    )
    lines.append(
        f"| 字段级准确率 (容差 ±{config.pipeline.value_tolerance:.0%}) | "
        f"**{summary.field_accuracy:.1%}** | {summary.field_hit}/{summary.field_total} |"
    )
    lines.append(
        f"| 行精确率 / 召回率 | {summary.row_precision:.1%} / {summary.row_recall:.1%} | |"
    )
    lines.append(
        f"| **硬给率**（应拒答却给了值） | **{summary.hard_give_rate:.1%}** | "
        f"{summary.hard_give_total}/{summary.expected_abstain_total} —— 越低越好 |"
    )
    lines.append(
        f"| 正确拒答率 | {summary.correct_abstain_rate:.1%} | "
        f"{summary.correct_abstain_total}/{summary.expected_abstain_total} |"
    )
    lines.append(
        f"| 硬给文档数（未拒答但准确率 < {config.pipeline.min_field_accuracy_to_pass:.0%}） | "
        f"**{summary.wrong_but_confident_docs}** | 越少越好 |"
    )
    lines.append("")
    lines.append("## 2. 逐文档明细")
    lines.append("")
    lines.append(
        "| 文档 | 状态 | 最终得分 | 轮次 | 抽取源 | 字段准确率 | 行P | 行R | 正确拒答 | 硬给 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for doc in summary.per_doc or []:
        flag = " ⚠" if doc.is_wrong_but_confident else ""
        lines.append(
            f"| {doc.pdf}{flag} | {doc.status} | {doc.final_score:.1f} | {doc.rounds} | "
            f"{doc.extraction_source} | {doc.field_accuracy:.1%} | "
            f"{doc.row_precision:.1%} | {doc.row_recall:.1%} | "
            f"{doc.correct_abstain} | {doc.hard_give} |"
        )
    lines.append("")

    lines.append("## 3. 拒答与硬给明细")
    lines.append("")
    any_detail = False
    for run, evaluation in pairs:
        if evaluation is None:
            continue
        if not (evaluation.hard_give or evaluation.correct_abstain or evaluation.abstained_rows):
            continue
        any_detail = True
        lines.append(f"### {run.pdf_name}")
        lines.append(f"- 文档状态：`{run.status}`；说明：{run.note or '-'}")
        if run.abstain_reasons:
            for reason in run.abstain_reasons:
                lines.append(f"  - 拒答原因：{reason}")
        if evaluation.correct_abstain:
            lines.append(f"- ✅ 正确拒答：{'; '.join(evaluation.correct_abstain)}")
        if evaluation.hard_give:
            lines.append(f"- ❌ **硬给**（应拒答却输出数值）：{'; '.join(evaluation.hard_give)}")
        if evaluation.abstained_rows:
            lines.append(f"- ⚠️ 过度拒答：{'; '.join(evaluation.abstained_rows)}")
        if evaluation.missing_rows:
            lines.append(f"- 缺失行：{'; '.join(evaluation.missing_rows)}")
        lines.append("")
    if not any_detail:
        lines.append("（无拒答或硬给记录）")
        lines.append("")

    lines.append("## 4. 迭代过程")
    lines.append("")
    for run, _ in pairs:
        lines.append(f"### {run.pdf_name}")
        for record in run.rounds:
            lines.append(
                f"- 第 {record.round_index} 轮：得分 {record.score:.1f}，裁决 `{record.verdict}`，"
                f"行数 {len(record.rows)}"
                + (f"，行级拒答 {record.abstain_row_indices}" if record.abstain_row_indices else "")
                + f" —— {record.summary}"
            )
            critical = [i for i in record.issues if i.severity == "critical"]
            for issue in critical[:5]:
                row = "整表" if issue.row_index is None else f"第 {issue.row_index} 行"
                lines.append(f"    - [critical] {row} / {issue.field}：{issue.message}")
        lines.append("")

    return "\n".join(lines)
