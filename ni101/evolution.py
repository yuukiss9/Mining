"""Evolution Log —— 失败/降级案例自动落盘，并可回放做 few-shot 改进。

题目原文：「每次失败/降级 case 自动 append 到 evolution.jsonl，题目最后复跑这个 log，
看你是否能用它做 few-shot 改进」。

所以这里有两个方向：

1. **写**：:func:`should_log` 判定"算不算失败/降级"，:func:`append_case` 追加一行 JSON。
2. **读**：:func:`build_few_shots` 把历史失败案例转成 Extractor 可用的 few-shot 示例，
   由 ``run_pipeline.py --replay-evolution`` 注入 prompt。

判定口径写在一处（:data:`_LOG_TRIGGERS`），别在别处复制——改一处漏一处是这类日志最常见的坑。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from .config import AppConfig
from .matcher import EvaluationResult
from .pipeline import STATUS_ABSTAIN, DocumentRun
from .schema import GroundTruthRow, ResourceRow

#: 触发写入 evolution.jsonl 的条件（任一命中即记录）
_LOG_TRIGGERS: tuple[str, ...] = (
    "document_abstained",       # 整篇拒答
    "partial_abstain",          # 有行级拒答
    "wrong_but_confident",      # 未拒答但准确率不达标（硬给）
    "hard_give",                # 应拒答的行被硬给了数值
    "extraction_fallback",      # 走了规则兜底（降级）
    "extraction_error",         # 抽取过程报错
    "rounds_exhausted",         # 迭代轮次耗尽
)


def _row_dump(row: ResourceRow | GroundTruthRow) -> dict[str, Any]:
    data = row.model_dump(mode="json")
    data.pop("abstain_reason", None)
    return data


def _index_rows(rows: Sequence[Any]) -> dict[tuple[str, str], Any]:
    index: dict[tuple[str, str], Any] = {}
    for row in rows:
        index.setdefault(row.identity_key(), row)
    return index


def should_log(
    run: DocumentRun,
    evaluation: Optional[EvaluationResult],
    config: AppConfig,
) -> tuple[bool, list[str]]:
    """返回 (是否记录, 命中的触发条件列表)。"""
    reasons: list[str] = []

    if run.status == STATUS_ABSTAIN:
        reasons.append("document_abstained")
    if any(row.abstain for row in run.final_rows):
        reasons.append("partial_abstain")
    if run.note.startswith("exhausted_"):
        reasons.append("rounds_exhausted")
    if run.extraction_source == "rule" or "fallback" in run.note:
        reasons.append("extraction_fallback")
    if any(record.error for record in run.rounds):
        reasons.append("extraction_error")

    if evaluation is not None:
        if evaluation.hard_give:
            reasons.append("hard_give")
        if (
            run.status != STATUS_ABSTAIN
            and evaluation.accuracy_denominator_rows > 0
            and evaluation.field_accuracy < config.pipeline.min_field_accuracy_to_pass
        ):
            reasons.append("wrong_but_confident")

    # 去重但保持 _LOG_TRIGGERS 的顺序，便于报告稳定
    ordered = [t for t in _LOG_TRIGGERS if t in reasons]
    return bool(ordered), ordered


def make_case(
    run: DocumentRun,
    evaluation: Optional[EvaluationResult],
    truth_rows: Sequence[GroundTruthRow] = (),
    triggers: Sequence[str] = (),
) -> dict[str, Any]:
    """构造一条 evolution 记录。

    ``predicted`` / ``corrected`` 是给 few-shot 用的：前者是错的预测行，
    后者是原文应有的行。两者都有内容时这条记录才有 few-shot 价值。
    """
    wrong_keys = {
        comparison.row_key
        for comparison in (evaluation.field_comparisons if evaluation else [])
        if not comparison.matched
    }
    pred_index = _index_rows(run.final_rows)
    truth_index = _index_rows(truth_rows)

    predicted: list[dict[str, Any]] = []
    for key in wrong_keys:
        row = pred_index.get(tuple(key.split("/", 1)))  # type: ignore[arg-type]
        if row is not None:
            predicted.append(_row_dump(row))
    if evaluation is not None:
        for extra in evaluation.extra_rows:
            for row in run.final_rows:
                if row.summary() == extra:
                    predicted.append(_row_dump(row))
                    break

    corrected: list[dict[str, Any]] = []
    for key in wrong_keys:
        row = truth_index.get(tuple(key.split("/", 1)))  # type: ignore[arg-type]
        if row is not None:
            corrected.append(_row_dump(row))

    if evaluation is not None and evaluation.hard_give:
        for label in evaluation.hard_give:
            key = tuple(label.split("/", 1))
            row = truth_index.get(key)  # type: ignore[arg-type]
            if row is not None and all(_row_dump(row) != item for item in corrected):
                corrected.append(_row_dump(row))

    reason_parts: list[str] = []
    if run.status == STATUS_ABSTAIN:
        reason_parts.append(f"文档级拒答：{'；'.join(run.abstain_reasons) or run.note}")
    if evaluation is not None and evaluation.hard_give:
        reason_parts.append(
            "下列行原文自相矛盾，应当拒答而不要硬给：" + "；".join(evaluation.hard_give)
        )
    if evaluation is not None and wrong_keys:
        reason_parts.append("以下行的字段与原文不符：" + "；".join(sorted(wrong_keys)))
    if not reason_parts:
        reason_parts.append("；".join(triggers) or run.note or "未通过审核")

    return {
        "case_id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pdf": run.pdf_name,
        "status": run.status,
        "triggers": list(triggers),
        "rounds": len(run.rounds),
        "final_score": round(run.final_score, 2),
        "reason": " | ".join(reason_parts),
        "abstain_reasons": list(run.abstain_reasons),
        "predicted": predicted,
        "corrected": corrected,
        "field_errors": (
            [c.to_dict() for c in evaluation.field_comparisons if not c.matched]
            if evaluation
            else []
        ),
    }


def append_case(path: Path, case: dict[str, Any]) -> Path:
    """把一条案例追加到 JSONL。目录不存在时自动创建。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    return path


def load_cases(path: Path) -> list[dict[str, Any]]:
    """读回所有案例。单行损坏时跳过而不是整份失败。"""
    path = Path(path)
    if not path.exists():
        return []
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return cases


def build_few_shots(
    cases: Sequence[dict[str, Any]],
    limit: int = 3,
    exclude_pdf: Optional[str] = None,
) -> list[dict[str, Any]]:
    """挑出可用于 few-shot 的历史案例（同一份 PDF 的案例排除，避免"抄答案"）。

    优先挑选 ``predicted`` 与 ``corrected`` 都非空的案例——这种才是"错误的输出
    + 正确的输出"的可学习对照。
    """
    usable = [
        case
        for case in cases
        if case.get("predicted") and case.get("corrected")
        and (exclude_pdf is None or case.get("pdf") != exclude_pdf)
    ]
    usable.sort(key=lambda c: c.get("created_at", ""), reverse=True)
    return [
        {
            "pdf": case.get("pdf", ""),
            "reason": case.get("reason", ""),
            "predicted": case.get("predicted", []),
            "corrected": case.get("corrected", []),
        }
        for case in usable[: max(0, limit)]
    ]


def truncate(path: Path) -> None:
    """清空日志（仅测试与演示复位使用）。"""
    path = Path(path)
    if path.exists():
        path.write_text("", encoding="utf-8")
