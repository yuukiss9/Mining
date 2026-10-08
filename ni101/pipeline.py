"""Revise Loop 编排 —— 题目要求的"<= 3 轮迭代，评分 >= 8 通过，否则 abstain"。

单份文档的流程::

    Extractor -> CriticMaster -> (未通过且还有轮次) 带反馈重抽 -> ...
                              -> 通过 / 轮次耗尽 -> 拒答 + 标注待人工审核

两个容易踩的坑，这里显式处理了：

1. **规则路径不空转**。未配置模型时抽取是确定性的，喂回反馈也不会变，
   此时不再重跑第 2、3 轮（那只会浪费时间并伪造"迭代过"的假象）。
2. **行级拒答先于打分**。原文自相矛盾的行会被移出"我承诺正确"的范围，
   避免一行坏数据把整份文档拖进全篇拒答，白丢正确的行。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .config import AppConfig
from .critic import CriticIssue, CriticMasterAgent, CriticVerdict
from .extractor import ExtractorAgent
from .pdf import PdfDocument, discover_pdfs, load_pdf
from .schema import ResourceRow

# 文档级状态
STATUS_PASS = "pass"            # 全部通过
STATUS_PARTIAL = "partial"      # 有行级拒答，其余通过（部分转人工）
STATUS_ABSTAIN = "abstain"      # 整体拒答，转人工审核

Logger = Callable[[str], None]


def _noop(_: str) -> None:
    return None


@dataclass
class RoundRecord:
    """一轮迭代的快照。"""

    round_index: int
    rows: list[ResourceRow] = field(default_factory=list)
    score: float = 0.0
    verdict: str = ""
    issues: list[CriticIssue] = field(default_factory=list)
    abstain_row_indices: list[int] = field(default_factory=list)
    critic_source: str = ""
    summary: str = ""
    extraction_note: str = ""
    latency_ms: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "score": round(self.score, 2),
            "verdict": self.verdict,
            "abstain_row_indices": self.abstain_row_indices,
            "critic_source": self.critic_source,
            "summary": self.summary,
            "extraction_note": self.extraction_note,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "rows": [row.model_dump(mode="json") for row in self.rows],
            "issues": [issue.model_dump(mode="json") for issue in self.issues],
        }


@dataclass
class DocumentRun:
    """一份 PDF 的完整运行结果。"""

    pdf_name: str
    status: str = STATUS_ABSTAIN
    rounds: list[RoundRecord] = field(default_factory=list)
    final_rows: list[ResourceRow] = field(default_factory=list)
    abstain: bool = False
    abstain_reasons: list[str] = field(default_factory=list)
    extraction_source: str = "unknown"
    final_score: float = 0.0
    note: str = ""
    pdf_warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0
    evaluation: Optional[dict[str, Any]] = None

    @property
    def pdf_stem(self) -> str:
        return Path(self.pdf_name).stem

    def to_dict(self) -> dict[str, Any]:
        return {
            "pdf": self.pdf_name,
            "status": self.status,
            "abstain": self.abstain,
            "abstain_reasons": self.abstain_reasons,
            "extraction_source": self.extraction_source,
            "final_score": round(self.final_score, 2),
            "note": self.note,
            "meta": self.meta,
            "pdf_warnings": self.pdf_warnings,
            "elapsed_ms": self.elapsed_ms,
            "rounds": [r.to_dict() for r in self.rounds],
            "final_rows": [row.model_dump(mode="json") for row in self.final_rows],
            "evaluation": self.evaluation,
        }


def _apply_row_abstains(
    rows: Sequence[ResourceRow], verdict: CriticVerdict
) -> list[str]:
    """把 Critic 标出的行标记为拒答，返回人类可读的拒绝原因。"""
    reasons: list[str] = []
    reason_by_index: dict[int, str] = {}
    for issue in verdict.issues:
        if issue.require_abstain and issue.row_index is not None:
            reason_by_index.setdefault(issue.row_index, issue.message)

    for index in verdict.abstain_row_indices:
        if not (0 <= index < len(rows)):
            continue
        row = rows[index]
        reason = reason_by_index.get(
            index, "CriticMaster 判定该行无法可靠确认，转人工审核。"
        )
        row.abstain = True
        row.abstain_reason = reason
        resource_class, commodity = row.identity_key()
        reasons.append(f"第 {index} 行（{resource_class} / {commodity}）拒答：{reason}")
    return reasons


def _merge_reasons(primary: Sequence[str], secondary: Sequence[str]) -> list[str]:
    """合并拒答原因并去重。

    行级拒答的原因与文档级 abstain_reasons 常常指向同一条 Critic 结论，
    不去重会在报告里出现两遍，读起来像是两个问题。
    """
    merged = list(primary)
    for item in secondary:
        if any(item[:60] in existing for existing in merged):
            continue
        merged.append(item)
    return merged


def run_document(
    pdf_path: Path | PdfDocument,
    config: AppConfig,
    extractor: ExtractorAgent,
    critic: CriticMasterAgent,
    few_shots: Optional[Sequence[dict[str, Any]]] = None,
    document: Optional[PdfDocument] = None,
    log: Logger = _noop,
) -> DocumentRun:
    """跑完一份 PDF 的 Revise Loop。

    ``pdf_path`` 也可以直接传一个已经读好的 :class:`~ni101.pdf.PdfDocument`，
    省掉重复解析（跑批时很有用）。
    """
    started = time.perf_counter()
    if isinstance(pdf_path, PdfDocument):
        document = pdf_path if document is None else document
        pdf_path = pdf_path.path
    pdf_path = Path(pdf_path)
    run = DocumentRun(pdf_name=pdf_path.name)

    try:
        doc = document if document is not None else load_pdf(pdf_path)
    except Exception as exc:  # noqa: BLE001
        run.status = STATUS_ABSTAIN
        run.abstain = True
        run.abstain_reasons = [f"PDF 读取失败：{exc}"]
        run.note = "pdf_read_failed"
        run.elapsed_ms = int((time.perf_counter() - started) * 1000)
        log(f"  [!] {pdf_path.name} 读取失败：{exc}")
        return run

    run.pdf_warnings = list(doc.warnings)
    run.meta = {
        "page_count": doc.page_count,
        "table_candidates": len(doc.candidate_tables()),
        "text_length": doc.text_length,
        "backend": doc.backend,
        **doc.extract_meta(),
    }

    if doc.is_probably_scanned:
        run.status = STATUS_ABSTAIN
        run.abstain = True
        run.abstain_reasons = [
            "疑似扫描件（无文本层）。本项目未实现 OCR，拒绝硬给结果。"
        ]
        run.note = "scanned_pdf"
        run.elapsed_ms = int((time.perf_counter() - started) * 1000)
        log("  [!] 疑似扫描件，直接拒答（未实现 OCR）")
        return run

    run.extraction_source = extractor.source
    feedback: Optional[list[dict[str, Any]]] = None
    abstain_reasons: list[str] = []

    for round_index in range(1, config.pipeline.max_rounds + 1):
        round_started = time.perf_counter()
        result = extractor.extract(doc, feedback=feedback, few_shots=few_shots)
        rows = list(result.rows)

        if result.abstain and not rows:
            verdict = CriticVerdict(
                score=1.0,
                verdict="abstain",
                summary="Extractor 主动拒答：" + "；".join(result.abstain_reasons),
                source=result.source,
                hard_abstain=True,
                abstain_reasons=list(result.abstain_reasons),
            )
        else:
            verdict = critic.review(doc, rows)

        row_reasons = _apply_row_abstains(rows, verdict)
        abstain_reasons = _merge_reasons(row_reasons, verdict.abstain_reasons)

        record = RoundRecord(
            round_index=round_index,
            rows=[row.model_copy(deep=True) for row in rows],
            score=verdict.score,
            verdict=verdict.verdict,
            issues=list(verdict.issues),
            abstain_row_indices=list(verdict.abstain_row_indices),
            critic_source=verdict.source,
            summary=verdict.summary,
            extraction_note=result.note,
            latency_ms=int((time.perf_counter() - round_started) * 1000),
            error=result.error or verdict.llm_error,
        )
        run.rounds.append(record)
        run.final_rows = rows
        run.final_score = verdict.score

        log(
            f"  · 第 {round_index} 轮 | 抽取源={result.source} | "
            f"行数={len(rows)} | Critic={verdict.score:.1f} | {verdict.verdict}"
            + (f" | 行级拒答 {len(verdict.abstain_row_indices)}" if verdict.abstain_row_indices else "")
        )

        if verdict.hard_abstain:
            run.status = STATUS_ABSTAIN
            run.abstain = True
            run.note = "critic_hard_abstain"
            break

        if verdict.score >= config.pipeline.pass_score:
            run.status = STATUS_PARTIAL if verdict.abstain_row_indices else STATUS_PASS
            run.abstain = False
            run.note = "passed"
            break

        # 未通过：判断是否值得再来一轮
        if not extractor.can_revise:
            run.status = STATUS_ABSTAIN
            run.abstain = True
            run.note = "no_revision_capability"
            abstain_reasons = abstain_reasons or [
                f"得分 {verdict.score:.1f} 未达 {config.pipeline.pass_score}，"
                "且当前抽取路径为确定性规则（无模型可返工），转人工审核。"
            ]
            break

        feedback = [issue.model_dump() for issue in verdict.issues]
        if round_index == config.pipeline.max_rounds:
            run.status = STATUS_ABSTAIN
            run.abstain = True
            run.note = f"exhausted_{config.pipeline.max_rounds}_rounds"
            abstain_reasons = abstain_reasons or [
                f"已迭代 {config.pipeline.max_rounds} 轮仍低于 {config.pipeline.pass_score} 分，转人工审核。"
            ]

    run.abstain_reasons = abstain_reasons
    run.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return run


def run_all(
    config: AppConfig,
    extractor: ExtractorAgent,
    critic: CriticMasterAgent,
    pdf_paths: Optional[Sequence[Path]] = None,
    few_shots: Optional[Sequence[dict[str, Any]]] = None,
    log: Logger = _noop,
) -> list[DocumentRun]:
    """跑批：默认扫描 ``config.pdf_dir`` 下所有 PDF（按文件名排序，保证可复现）。"""
    paths = list(pdf_paths) if pdf_paths else discover_pdfs(config.pdf_dir)
    runs: list[DocumentRun] = []
    for path in paths:
        log(f"[run] {Path(path).name}")
        runs.append(
            run_document(
                Path(path),
                config,
                extractor,
                critic,
                few_shots=few_shots,
                log=log,
            )
        )
    return runs
