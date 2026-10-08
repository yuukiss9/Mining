"""Extractor Agent —— 从 NI 43-101 PDF 中抽取资源量表。

两条路径，行为对等：

* **LLM 路径**（配置了 api_key / base_url）：把定位到的候选表与正文片段交给强模型，
  按 :mod:`ni101.prompt` 里定义的结构输出。
* **规则路径**（未配置 / LLM 调用失败）：用确定性解析从表格结构里读数，支持
  多矿种并列表（如 Barrick 的 Cu 与 Au 同表）。

规则路径不是"假数据"——它读的是 PDF 里真实存在的表格，只是鲁棒性弱于模型。
它存在的意义是让配置 key 之前整条链路就能跑通。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .llm import BaseLLM
from .pdf import PdfDocument, PdfTable
from .prompt import EXTRACTOR_SYSTEM, build_extractor_user
from .schema import (
    ExtractionResult,
    GradeUnit,
    MetalUnit,
    ResourceClass,
    ResourceRow,
    parse_number,
)

# --------------------------------------------------------------------------
# 表头识别用正则
# --------------------------------------------------------------------------
_ELEMENT_RE = re.compile(
    r"\b(Au|Ag|Cu|Pb|Zn|Li2O|Li|Fe2O3|Fe|Mo|Ni|Co|Sn|U3O8|WO3|Mn|Al2O3|PGM|Pt|Pd|"
    r"Nb2O5|Ta2O5|TREO|REO)\b",
    re.IGNORECASE,
)
_TONNAGE_HINT = re.compile(r"(tonnes?|tonnage|quantity|\bmt\b|\bkt\b)", re.IGNORECASE)
_METAL_HINT = re.compile(r"(contained|metal|ounces?|\boz\b|\bkoz\b|pounds?|\blb\b)", re.IGNORECASE)
_GRADE_HINT = re.compile(r"(grade|g/t|gpt|\bppm\b|%)", re.IGNORECASE)
#: 合计行 / 小计行标签，必须跳过，否则会被当成一个资源类别重复计入
_SKIP_LABEL = re.compile(
    r"^\s*(total|sub-?total|combined|aggregate|measured\s*&\s*indicated|"
    r"indicated\s*&\s*inferred|m\s*&\s*i)\b",
    re.IGNORECASE,
)

_ELEMENT_ALIASES = {
    "AU": "Au", "AG": "Ag", "CU": "Cu", "PB": "Pb", "ZN": "Zn", "FE": "Fe",
    "MO": "Mo", "NI": "Ni", "CO": "Co", "SN": "Sn", "LI": "Li2O", "LI2O": "Li2O",
    "FE2O3": "Fe2O3", "U3O8": "U3O8", "WO3": "WO3", "MN": "Mn", "AL2O3": "Al2O3",
    "PGM": "PGM", "PT": "Pt", "PD": "Pd", "NB2O5": "Nb2O5", "TA2O5": "Ta2O5",
    "TREO": "TREO", "REO": "REO",
}


def normalize_element(raw: str) -> str:
    text = (raw or "").strip()
    return _ELEMENT_ALIASES.get(text.upper(), text) if text else ""


def _detect_element(*texts: str) -> str:
    for text in texts:
        if not text:
            continue
        match = _ELEMENT_RE.search(text)
        if match:
            return normalize_element(match.group(1))
    return ""


# --------------------------------------------------------------------------
# 列布局识别
# --------------------------------------------------------------------------
@dataclass
class _ColumnSpec:
    index: int
    unit: Any = None
    element: str = ""


@dataclass
class _TableLayout:
    header_index: int
    header_cells: list[str]
    tonnage: Optional[_ColumnSpec] = None
    grades: list[_ColumnSpec] = field(default_factory=list)
    metals: list[_ColumnSpec] = field(default_factory=list)

    @property
    def hits(self) -> int:
        return (1 if self.tonnage else 0) + len(self.grades) + len(self.metals)


def _detect_layout(table: PdfTable) -> Optional[_TableLayout]:
    """在前 5 行里找表头行，并识别 吨位列 / 品位列(可多) / 金属量列(可多)。"""
    best: Optional[_TableLayout] = None

    for index, row in enumerate(table.rows[:5]):
        layout = _TableLayout(header_index=index, header_cells=list(row))
        for col, cell in enumerate(row):
            text = (cell or "").strip()
            if not text:
                continue
            element = _detect_element(text)

            grade_unit = GradeUnit.parse(text)
            metal_unit = MetalUnit.parse(text)

            if grade_unit is not None and _GRADE_HINT.search(text):
                layout.grades.append(_ColumnSpec(col, grade_unit, element))
                continue
            if metal_unit is not None and _METAL_HINT.search(text):
                layout.metals.append(_ColumnSpec(col, metal_unit, element))
                continue
            if _TONNAGE_HINT.search(text) and layout.tonnage is None:
                layout.tonnage = _ColumnSpec(col)
                continue

        if best is None or layout.hits > best.hits:
            best = layout

    if best is None or best.hits < 2:
        return None
    return best


def _pair_grade_metal(
    grades: list[_ColumnSpec], metals: list[_ColumnSpec]
) -> list[tuple[_ColumnSpec, Optional[_ColumnSpec]]]:
    """把品位列与金属量列配对：优先同矿种，其次按位置顺次占用。"""
    pairs: list[tuple[_ColumnSpec, Optional[_ColumnSpec]]] = []
    used: set[int] = set()
    for grade in grades:
        match: Optional[_ColumnSpec] = None
        if grade.element:
            for metal in metals:
                if metal.index in used:
                    continue
                if metal.element == grade.element:
                    match = metal
                    break
        if match is None:
            for metal in metals:
                if metal.index not in used:
                    match = metal
                    break
        if match is not None:
            used.add(match.index)
        pairs.append((grade, match))
    return pairs


class ExtractorAgent:
    """Extractor Agent：LLM 优先，失败自动落到确定性规则解析。"""

    def __init__(self, llm: BaseLLM, verbose: bool = False) -> None:
        self.llm = llm
        self.verbose = verbose
        self.last_error: Optional[str] = None

    @property
    def source(self) -> str:
        return "llm" if self.llm.available else "rule"

    @property
    def can_revise(self) -> bool:
        """规则路径是确定性的，喂回反馈也不会变——避免 Revise Loop 空转。"""
        return self.llm.available

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def extract(
        self,
        document: PdfDocument,
        feedback: Optional[Sequence[dict[str, Any]]] = None,
        few_shots: Optional[Sequence[dict[str, Any]]] = None,
    ) -> ExtractionResult:
        if document.is_probably_scanned:
            return ExtractionResult(
                rows=[],
                source=self.source,
                abstain=True,
                abstain_reasons=[
                    "疑似扫描件，无可用文本层；本项目未实现 OCR，拒绝硬给结果。"
                ],
                note="scanned_pdf",
            )

        if self.llm.available:
            result = self._extract_with_llm(document, feedback, few_shots)
            if result is not None and result.rows:
                return result
            fallback = self._extract_with_rules(document)
            if result is not None and result.error:
                fallback.error = result.error
                fallback.note = f"llm_failed_fallback_rule ({result.error})"
            elif result is not None and result.abstain:
                fallback.abstain = True
                fallback.abstain_reasons = result.abstain_reasons or fallback.abstain_reasons
                fallback.note = "llm_abstain"
            return fallback

        return self._extract_with_rules(document)

    # ------------------------------------------------------------------
    # LLM 路径
    # ------------------------------------------------------------------
    def _extract_with_llm(
        self,
        document: PdfDocument,
        feedback: Optional[Sequence[dict[str, Any]]],
        few_shots: Optional[Sequence[dict[str, Any]]],
    ) -> Optional[ExtractionResult]:
        context = document.context_for_llm()
        user = build_extractor_user(context, feedback=feedback, few_shots=few_shots)
        try:
            raw, meta = self.llm.complete_json(EXTRACTOR_SYSTEM, user, json_mode=True)
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            if self.verbose:
                print(f"[extractor] LLM 抽取失败：{exc}")
            return ExtractionResult(
                rows=[], source="llm", error=str(exc), note="llm_call_failed"
            )

        if isinstance(raw, dict) and "rows" in raw:
            payload = raw
        elif isinstance(raw, list):
            payload = {"rows": raw}
        else:
            error = "模型输出缺少 rows 字段"
            self.last_error = error
            return ExtractionResult(rows=[], source="llm", error=error, note="bad_schema")

        rows: list[ResourceRow] = []
        for item in payload.get("rows") or []:
            if not isinstance(item, dict):
                continue
            try:
                row = ResourceRow(**item)
            except Exception:  # noqa: BLE001 - 单行格式错误不该拖垮整份文档
                continue
            if self._is_empty_row(row):
                continue
            rows.append(row)

        return ExtractionResult(
            rows=rows,
            source="llm",
            abstain=bool(payload.get("abstain", False)),
            abstain_reasons=list(payload.get("abstain_reasons") or []),
            note=f"model={meta.model} latency={meta.latency_ms}ms",
        )

    @staticmethod
    def _is_empty_row(row: ResourceRow) -> bool:
        return (
            row.resource_class is None
            and not row.commodity.strip()
            and row.tonnage_mt is None
            and row.grade is None
            and row.contained_metal is None
        )

    # ------------------------------------------------------------------
    # 规则路径
    # ------------------------------------------------------------------
    def _extract_with_rules(self, document: PdfDocument) -> ExtractionResult:
        tables = document.candidate_tables(top_k=4)
        collected: dict[tuple[str, str], ResourceRow] = {}
        tables_used = 0

        for table in tables:
            rows = self._rows_from_table(table)
            if rows:
                tables_used += 1
            for row in rows:
                key = row.identity_key()
                existing = collected.get(key)
                if existing is not None and existing.is_complete:
                    continue
                collected[key] = row

        if not collected:
            for page_no, text in enumerate(document.page_texts, start=1):
                for row in self._rows_from_text(text, page_no):
                    collected.setdefault(row.identity_key(), row)

        rows = list(collected.values())
        if not rows:
            return ExtractionResult(
                rows=[],
                source="rule",
                abstain=True,
                abstain_reasons=[
                    "规则路径未能从 PDF 中定位到可读的资源量表；"
                    "规则路径仅作兜底，建议配置真实模型后重跑。"
                ],
                note="rule_no_table_found",
            )

        return ExtractionResult(
            rows=rows,
            source="rule",
            note=f"tables_used={tables_used}（确定性规则解析，未经模型语义理解）",
        )

    # -- 表 -> 行 --------------------------------------------------------
    def _rows_from_table(self, table: PdfTable) -> list[ResourceRow]:
        if not table.rows:
            return []

        layout = _detect_layout(table)
        if layout is None:
            return self._rows_from_positional(table)

        header_cells = layout.header_cells
        rows: list[ResourceRow] = []
        for raw_row in table.rows[layout.header_index + 1 :]:
            if not raw_row:
                continue
            row_text = " ".join(c for c in raw_row if c).strip()
            if not row_text:
                continue

            resource_class = self._detect_class(raw_row)
            if resource_class is None:
                continue  # 合计行 / 小计行 / 标题行

            tonnage = self._cell_number(raw_row, layout.tonnage.index if layout.tonnage else None)
            pairs = _pair_grade_metal(layout.grades, layout.metals)

            if not pairs:
                # 只认出吨位（少见）——保留一行，其余字段留空由 Critic 判定
                if tonnage is not None:
                    rows.append(
                        self._make_row(
                            resource_class, "", tonnage, None, None, None, None,
                            table.page_number, row_text,
                        )
                    )
                continue

            for grade_spec, metal_spec in pairs:
                grade = self._cell_number(raw_row, grade_spec.index)
                metal = self._cell_number(raw_row, metal_spec.index if metal_spec else None)
                if tonnage is None and grade is None and metal is None:
                    continue
                commodity = grade_spec.element or (metal_spec.element if metal_spec else "")
                if not commodity:
                    commodity = _detect_element(row_text)
                rows.append(
                    self._make_row(
                        resource_class,
                        commodity,
                        tonnage,
                        grade,
                        grade_spec.unit,
                        metal,
                        metal_spec.unit if metal_spec else None,
                        table.page_number,
                        row_text,
                    )
                )
        return rows

    @staticmethod
    def _make_row(
        resource_class: ResourceClass,
        commodity: str,
        tonnage: Optional[float],
        grade: Optional[float],
        grade_unit: Optional[GradeUnit],
        metal: Optional[float],
        metal_unit: Optional[MetalUnit],
        page_number: int,
        row_text: str,
    ) -> ResourceRow:
        return ResourceRow(
            resource_class=resource_class,
            commodity=commodity or "unknown",
            tonnage_mt=tonnage,
            grade=grade,
            grade_unit=grade_unit,
            contained_metal=metal,
            contained_metal_unit=metal_unit,
            source_page=page_number,
            source_snippet=row_text[:200],
            confidence=0.6,
        )

    def _rows_from_positional(self, table: PdfTable) -> list[ResourceRow]:
        """表头完全识别不出时的位置启发式：类别 + 紧跟的若干数字。"""
        rows: list[ResourceRow] = []
        for raw_row in table.rows:
            resource_class = self._detect_class(raw_row)
            if resource_class is None:
                continue
            numeric = [parse_number(c) for c in raw_row]
            numbers = [v for v in numeric if v is not None]
            if len(numbers) < 3:
                continue
            row_text = " ".join(c for c in raw_row if c).strip()
            rows.append(
                self._make_row(
                    resource_class,
                    _detect_element(row_text),
                    numbers[0],
                    numbers[1],
                    GradeUnit.parse(row_text),
                    numbers[2],
                    MetalUnit.parse(row_text),
                    table.page_number,
                    row_text,
                )
            )
        return rows

    @staticmethod
    def _cell_number(row: list[str], column: Optional[int]) -> Optional[float]:
        if column is None or column >= len(row):
            return None
        return parse_number(row[column])

    @staticmethod
    def _detect_class(row: list[str]) -> Optional[ResourceClass]:
        label = " ".join(c for c in row[:2] if c).strip()
        if _SKIP_LABEL.match(label):
            return None  # 合计行 / 小计行
        for cell in row[:2]:
            parsed = ResourceClass.parse(cell)
            if parsed is not None:
                return parsed
        return None

    # -- 纯文本兜底 ------------------------------------------------------
    @staticmethod
    def _rows_from_text(text: str, page_number: int) -> list[ResourceRow]:
        """表格结构完全没识别出来时，从正文行里按 "类别 + 3 个数字" 兜底抽取。"""
        rows: list[ResourceRow] = []
        pattern = re.compile(
            r"^\s*(measured|indicated|inferred)\b[^\d\n]{0,40}"
            r"([\d,]+\.?\d*)\s+([\d,]+\.?\d*)\s+([\d,]+\.?\d*)",
            re.IGNORECASE,
        )
        for line in text.splitlines():
            match = pattern.search(line)
            if not match:
                continue
            resource_class = ResourceClass.parse(match.group(1))
            if resource_class is None:
                continue
            rows.append(
                ResourceRow(
                    resource_class=resource_class,
                    commodity=_detect_element(line) or "unknown",
                    tonnage_mt=parse_number(match.group(2)),
                    grade=parse_number(match.group(3)),
                    grade_unit=GradeUnit.parse(line),
                    contained_metal=parse_number(match.group(4)),
                    contained_metal_unit=MetalUnit.parse(line),
                    source_page=page_number,
                    source_snippet=line.strip()[:200],
                    confidence=0.4,
                )
            )
        return rows
