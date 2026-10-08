"""CriticMaster Agent —— 对抽取结果做对抗性审核并打分（1-10）。

两层审核叠加：

* **确定性规则层**（始终运行，永远是硬证据）：三角一致性、单位混淆、量级合理性、
  字段完整性、来源可追溯性、类别覆盖。这一层不看模型脸色。
* **语义审核层**（配置了 Critic 模型时运行）：看原文比对数值、识别行错位、
  读懂脚注对数值口径的影响。

最终得分取两层的**较低值**——规则层是我们对"明显错误"的兜底承诺，
不允许被语义层的宽松判断覆盖。
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .config import PipelineSettings
from .llm import BaseLLM
from .pdf import PdfDocument
from .prompt import CRITIC_SYSTEM, build_critic_user
from .schema import GradeUnit, ResourceClass, ResourceRow

Severity = Literal["critical", "major", "minor"]

_SEVERITY_PENALTY: dict[str, float] = {
    "critical": 3.0,
    "major": 1.5,
    "minor": 0.4,
}

# 量级合理性上限（超过即认为抽错，例如把品位当成吨位）
_MAX_TONNAGE_MT = 100_000.0
_MAX_GRADE_G_T = 10_000.0


class CriticIssue(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    row_index: Optional[int] = None
    field: str = ""
    severity: Severity = "major"
    issue_type: str = ""
    message: str = ""
    suggestion: str = ""
    require_abstain: bool = False


class CriticVerdict(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    score: float = 1.0
    verdict: Literal["pass", "revise", "abstain"] = "revise"
    summary: str = ""
    issues: list[CriticIssue] = Field(default_factory=list)
    source: str = "rule"
    #: 该行原文自相矛盾或无法确认 -> 应当行级拒答，交由人工审核
    abstain_row_indices: list[int] = Field(default_factory=list)
    #: 整篇无法可靠抽取 -> 文档级拒答
    hard_abstain: bool = False
    abstain_reasons: list[str] = Field(default_factory=list)
    llm_error: Optional[str] = None

    @property
    def critical_issues(self) -> list[CriticIssue]:
        return [i for i in self.issues if i.severity == "critical"]


class CriticMasterAgent:
    """CriticMaster：挑刺、打分、给出可执行的修改建议。"""

    def __init__(
        self,
        llm: BaseLLM,
        settings: PipelineSettings,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.settings = settings
        self.verbose = verbose
        self.last_error: Optional[str] = None

    @property
    def source(self) -> str:
        return "llm+rule" if self.llm.available else "rule"

    # ------------------------------------------------------------------
    def review(self, document: PdfDocument, rows: Sequence[ResourceRow]) -> CriticVerdict:
        rule_issues, abstain_indices, rule_score, rule_abstain = self._rule_review(
            document, rows
        )

        if not self.llm.available:
            return self._finalize(
                rule_score, rule_issues, abstain_indices, rule_abstain, source="rule"
            )

        llm_verdict = self._llm_review(document, rows, rule_issues)
        if llm_verdict is None:
            verdict = self._finalize(
                rule_score,
                rule_issues,
                abstain_indices,
                rule_abstain,
                source="rule(llm_failed)",
            )
            verdict.llm_error = self.last_error
            return verdict

        merged_issues = self._merge_issues(rule_issues, llm_verdict.issues)
        merged_abstain = sorted(set(abstain_indices) | set(llm_verdict.abstain_row_indices))
        # 分数取较低值：规则层是硬证据，不可被语义层的宽松判断盖过
        merged_score = min(rule_score, llm_verdict.score)

        verdict = self._finalize(
            merged_score,
            merged_issues,
            merged_abstain,
            rule_abstain or llm_verdict.hard_abstain,
            source="llm+rule",
        )
        if llm_verdict.summary:
            verdict.summary = f"{llm_verdict.summary}（规则层得分 {rule_score:.1f}）"
        verdict.llm_error = self.last_error
        return verdict

    # ------------------------------------------------------------------
    # 确定性规则层
    # ------------------------------------------------------------------
    def _rule_review(
        self, document: PdfDocument, rows: Sequence[ResourceRow]
    ) -> tuple[list[CriticIssue], list[int], float, bool]:
        issues: list[CriticIssue] = []

        if not rows:
            issues.append(
                CriticIssue(
                    row_index=None,
                    field="rows",
                    severity="critical",
                    issue_type="no_rows",
                    message="未抽取到任何资源量数据行。",
                    suggestion="确认 PDF 是否为扫描件、资源量表是否在文本层内。",
                    require_abstain=True,
                )
            )
            return issues, [], 3.0, True

        # ---- 第一遍：找出"原文自相矛盾"的行，这些行必须拒答 ----
        abstain_indices: list[int] = []
        for index, row in enumerate(rows):
            if row.abstain:
                continue  # 已经拒答的行不再重复惩罚
            deviation = row.deviation
            if deviation is not None and deviation > self.settings.consistency_hard_fail:
                abstain_indices.append(index)
                issues.append(
                    CriticIssue(
                        row_index=index,
                        field="contained_metal",
                        severity="critical",
                        issue_type="consistency_violation",
                        message=(
                            f"{row.summary()} —— 吨位 x 品位 推算出的金属量应为 "
                            f"{row.expected_metal_tonnes:,.0f} t，原文印刷值为 "
                            f"{row.metal_tonnes:,.0f} t，相对偏差 {deviation:.0%}，"
                            f"超出 {self.settings.consistency_hard_fail:.0%} 的硬阈值。"
                        ),
                        suggestion=(
                            "原文数值自相矛盾，无法判断哪个数字正确。"
                            "该行应行级拒答并标注待人工审核，禁止硬给结果。"
                        ),
                        require_abstain=True,
                    )
                )

        # ---- 第二遍：对所有行做常规校验 ----
        # 拒答的行同样要报出问题（排查时需要看到全部证据），
        # 只是它们的扣分会被排除在最终得分之外。
        abstain_set = set(abstain_indices)
        for index, row in enumerate(rows):
            if row.abstain:
                continue  # 已由上游明确拒答的行不再重复挑错
            issues.extend(self._check_row(index, row))

        issues.extend(self._check_document(document, rows))

        # ---- 打分：只统计未拒答行的扣分（拒答行已退出"我承诺正确"的范围）----
        scoring_issues = [
            issue
            for issue in issues
            if not issue.require_abstain and issue.row_index not in abstain_set
        ]
        score = 10.0
        for issue in scoring_issues:
            score -= _SEVERITY_PENALTY.get(issue.severity, 1.0)
        score = max(1.0, min(10.0, score))

        hard_abstain = len(abstain_indices) == len(rows) and len(rows) > 0
        if hard_abstain:
            score = min(score, 3.0)

        return issues, abstain_indices, score, hard_abstain

    def _check_row(self, index: int, row: ResourceRow) -> list[CriticIssue]:
        issues: list[CriticIssue] = []

        if row.resource_class is None:
            issues.append(
                CriticIssue(
                    row_index=index,
                    field="resource_class",
                    severity="critical",
                    issue_type="unclassified_row",
                    message=f"第 {index} 行没有识别出资源类别：{row.source_snippet or row.summary()}",
                    suggestion="该行可能是合计行或表头残留，应从结果中移除或改判为 Indicated/Inferred。",
                )
            )

        missing = [
            field
            for field in (
                "tonnage_mt",
                "grade",
                "grade_unit",
                "contained_metal",
                "contained_metal_unit",
            )
            if getattr(row, field) is None
        ]
        # 每个缺失字段单独记一条：这样得分能反映"缺了多少"，而不是一刀切。
        for field in missing:
            issues.append(
                CriticIssue(
                    row_index=index,
                    field=field,
                    severity="major",
                    issue_type="missing_field",
                    message=f"第 {index} 行缺失字段：{field}。",
                    suggestion="回到原文补齐；原文确实未印刷的保持 null，不要推算。",
                )
            )

        deviation = row.deviation
        if deviation is not None and self.settings.consistency_tolerance < deviation <= (
            self.settings.consistency_hard_fail
        ):
            issues.append(
                CriticIssue(
                    row_index=index,
                    field="contained_metal",
                    severity="critical",
                    issue_type="consistency_violation",
                    message=(
                        f"第 {index} 行三角一致性偏差 {deviation:.0%}"
                        f"（容差 {self.settings.consistency_tolerance:.0%}）。"
                    ),
                    suggestion="核对吨位、品位、金属量三列，通常是把 koz 当成 oz 或小数点错位。",
                )
            )

        if row.tonnage_mt is not None and not (0 < row.tonnage_mt <= _MAX_TONNAGE_MT):
            issues.append(
                CriticIssue(
                    row_index=index,
                    field="tonnage_mt",
                    severity="major",
                    issue_type="implausible_magnitude",
                    message=f"第 {index} 行吨位 {row.tonnage_mt} Mt 不在合理区间 (0, {_MAX_TONNAGE_MT:,.0f}]。",
                    suggestion="单位可能是 t 或 kt，未换算成 Mt。",
                )
            )

        if row.grade is not None:
            if row.grade <= 0:
                issues.append(
                    CriticIssue(
                        row_index=index,
                        field="grade",
                        severity="major",
                        issue_type="implausible_magnitude",
                        message=f"第 {index} 行品位 {row.grade} 非正数。",
                    )
                )
            elif row.grade_unit == GradeUnit.PERCENT and row.grade > 100:
                issues.append(
                    CriticIssue(
                        row_index=index,
                        field="grade",
                        severity="critical",
                        issue_type="unit_confusion",
                        message=f"第 {index} 行品位 {row.grade} % 超过 100%，单位大概率标错。",
                        suggestion="该数值可能是 g/t，请核对原文表头。",
                    )
                )
            elif row.grade_unit == GradeUnit.G_T and row.grade > _MAX_GRADE_G_T:
                issues.append(
                    CriticIssue(
                        row_index=index,
                        field="grade",
                        severity="major",
                        issue_type="implausible_magnitude",
                        message=f"第 {index} 行品位 {row.grade} g/t 量级异常。",
                    )
                )

        if row.contained_metal is not None and row.contained_metal <= 0:
            issues.append(
                CriticIssue(
                    row_index=index,
                    field="contained_metal",
                    severity="major",
                    issue_type="implausible_magnitude",
                    message=f"第 {index} 行金属量 {row.contained_metal} 非正数。",
                )
            )

        if row.source_page is None:
            issues.append(
                CriticIssue(
                    row_index=index,
                    field="source_page",
                    severity="minor",
                    issue_type="missing_provenance",
                    message=f"第 {index} 行缺少来源页码，无法追溯。",
                    suggestion="补上该行所在的 PDF 页码。",
                )
            )
        if not (row.source_snippet or "").strip():
            issues.append(
                CriticIssue(
                    row_index=index,
                    field="source_snippet",
                    severity="minor",
                    issue_type="missing_provenance",
                    message=f"第 {index} 行缺少原文片段，无法逐字核对。",
                )
            )

        return issues

    def _check_document(
        self, document: PdfDocument, rows: Sequence[ResourceRow]
    ) -> list[CriticIssue]:
        issues: list[CriticIssue] = []
        full_text = document.full_text.lower()

        present = {row.resource_class for row in rows if row.resource_class is not None}
        for resource_class in (ResourceClass.INDICATED, ResourceClass.INFERRED):
            if resource_class not in present and resource_class.value in full_text:
                issues.append(
                    CriticIssue(
                        row_index=None,
                        field="rows",
                        severity="major",
                        issue_type="coverage_gap",
                        message=(
                            f"原文中出现过 '{resource_class.value}'，但结果里没有任何该类别数据行，疑似漏抽。"
                        ),
                        suggestion="检查是否有跨页表格或该类别写在另一张表里。",
                    )
                )

        seen: dict[tuple[str, str], int] = {}
        for index, row in enumerate(rows):
            key = row.identity_key()
            if key in seen:
                issues.append(
                    CriticIssue(
                        row_index=index,
                        field="rows",
                        severity="major",
                        issue_type="duplicate_row",
                        message=f"第 {index} 行与第 {seen[key]} 行的 (类别, 矿种) 重复。",
                        suggestion="合并或删除重复行。",
                    )
                )
            else:
                seen[key] = index

        return issues

    # ------------------------------------------------------------------
    # 语义审核层
    # ------------------------------------------------------------------
    def _llm_review(
        self,
        document: PdfDocument,
        rows: Sequence[ResourceRow],
        rule_issues: Sequence[CriticIssue],
    ) -> Optional[CriticVerdict]:
        context = document.context_for_llm()
        rule_payload = [issue.model_dump() for issue in rule_issues]
        user = build_critic_user(context, rows, rule_payload)
        try:
            raw, _meta = self.llm.complete_json(CRITIC_SYSTEM, user, json_mode=True)
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            if self.verbose:
                print(f"[critic] LLM 审核失败：{exc}")
            return None

        if not isinstance(raw, dict):
            self.last_error = "模型输出不是 JSON 对象"
            return None

        issues: list[CriticIssue] = []
        for item in raw.get("issues") or []:
            if not isinstance(item, dict):
                continue
            severity = str(item.get("severity", "major")).lower()
            if severity not in ("critical", "major", "minor"):
                severity = "major"
            row_index = item.get("row_index")
            if isinstance(row_index, str) and row_index.strip().isdigit():
                row_index = int(row_index)
            elif not isinstance(row_index, int):
                row_index = None
            issues.append(
                CriticIssue(
                    row_index=row_index,
                    field=str(item.get("field", "")),
                    severity=severity,  # type: ignore[arg-type]
                    issue_type=str(item.get("issue_type", "llm_finding")),
                    message=str(item.get("message", "")),
                    suggestion=str(item.get("suggestion", "")),
                )
            )

        score = raw.get("score", 1.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 1.0
        score = max(1.0, min(10.0, score))

        abstain_rows: list[int] = []
        for value in raw.get("abstain_rows") or []:
            if isinstance(value, int):
                abstain_rows.append(value)
            elif isinstance(value, str) and value.strip().isdigit():
                abstain_rows.append(int(value))

        verdict_name = str(raw.get("verdict", "revise")).lower()
        if verdict_name not in ("pass", "revise", "abstain"):
            verdict_name = "revise"

        return CriticVerdict(
            score=score,
            verdict=verdict_name,  # type: ignore[arg-type]
            summary=str(raw.get("summary", "")),
            issues=issues,
            source="llm",
            abstain_row_indices=abstain_rows,
            hard_abstain=verdict_name == "abstain",
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _merge_issues(
        rule_issues: Sequence[CriticIssue], llm_issues: Sequence[CriticIssue]
    ) -> list[CriticIssue]:
        """规则问题优先保留（硬证据），LLM 问题去重后追加。"""
        merged: list[CriticIssue] = list(rule_issues)
        seen = {(i.row_index, i.field, i.issue_type) for i in rule_issues}
        for issue in llm_issues:
            key = (issue.row_index, issue.field, issue.issue_type)
            if key in seen:
                continue
            seen.add(key)
            merged.append(issue)
        return merged

    def _finalize(
        self,
        score: float,
        issues: Sequence[CriticIssue],
        abstain_rows: Sequence[int],
        hard_abstain: bool,
        source: str,
    ) -> CriticVerdict:
        if hard_abstain:
            verdict_name: Literal["pass", "revise", "abstain"] = "abstain"
        elif score >= self.settings.pass_score:
            verdict_name = "pass"
        else:
            verdict_name = "revise"

        reasons = [i.message for i in issues if i.require_abstain]
        if hard_abstain and not reasons:
            reasons = ["CriticMaster 判定无法可靠抽取，转人工审核。"]

        summary = {
            "pass": f"通过：得分 {score:.1f} >= {self.settings.pass_score}",
            "revise": f"需返工：得分 {score:.1f} < {self.settings.pass_score}",
            "abstain": "拒答：判定无法可靠抽取，转人工审核",
        }[verdict_name]
        if abstain_rows:
            summary += f"；另有 {len(abstain_rows)} 行行级拒答"

        return CriticVerdict(
            score=score,
            verdict=verdict_name,
            summary=summary,
            issues=list(issues),
            source=source,
            abstain_row_indices=sorted(set(abstain_rows)),
            hard_abstain=hard_abstain,
            abstain_reasons=reasons,
        )
