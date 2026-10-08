"""PDF 读取与"资源量表"定位。

真实 NI 43-101 报告动辄一两百页，绝大多数内容与资源量无关。
所以这里不是"把 PDF 喂给模型"，而是先做一次**候选表定位**：
只把命中的页与表送进 LLM 上下文，既控 token 也提准确率。

对外只暴露两个对象：:class:`PdfDocument` 与 :class:`PdfTable`。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .schema import parse_number

# --------------------------------------------------------------------------
# 关键词与权重（定位候选表的依据）
# --------------------------------------------------------------------------
#: 命中即强信号：出现这些词的页才值得抽表
PAGE_KEYWORDS = (
    "indicated",
    "inferred",
    "measured",
    "mineral resource",
    "resource estimate",
    "resource statement",
)

_STRONG_TABLE_TERMS = (
    "indicated",
    "inferred",
    "measured",
    "mineral resource",
    "resource estimate",
    "contained",
)
_MEDIUM_TABLE_TERMS = (
    "tonnes",
    "tonnage",
    "grade",
    "cut-off",
    "cutoff",
    "mt",
    "koz",
    "oz",
    "g/t",
    "%",
)

#: 表头里代表"单位"的单元格特征。
#: 注意：裸的 "t" **不能**算——"category"、"estimate" 里都有 t，
#: 放宽到裸字母会让正文碎片被误判成表头并拿到加分。
_UNIT_CELL = re.compile(
    r"(?<![a-z])(?:mt|kt|koz|moz|oz|lb|g/t|gpt|ppm|%)(?![a-z])",
    re.IGNORECASE,
)

#: 平均每页可提取文本少于该字符数，基本可判定为扫描件（无文本层）。
#: **公开常量**：``scripts/pdf_health.py`` 复用它做体检，保证"体检结论"与
#: "流水线是否拒答"同源 —— 两处阈值不一致会出现"体检说需要 OCR、流水线却照常解析"
#: 这种自相矛盾的结论。
SCANNED_CHARS_PER_PAGE = 30


# --------------------------------------------------------------------------
# 数据容器
# --------------------------------------------------------------------------
def _cell(value: Any) -> str:
    """单元格归一化：None -> ""，折叠空白与换行。"""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def render_table_markdown(rows: Iterable[Iterable[Any]]) -> str:
    """把二维表渲染成 markdown（喂给 LLM 的可读形式）。"""
    out: list[str] = []
    for row in rows:
        out.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return "\n".join(out)


@dataclass
class PdfTable:
    """PDF 中的一张表（由 pdfplumber 结构化出来的二维数组）。"""

    page_number: int          # 1 起
    index_on_page: int        # 该页第几张表，0 起
    rows: list[list[str]] = field(default_factory=list)
    score: float = 0.0

    @property
    def flat_text(self) -> str:
        return " ".join(_cell(c) for row in self.rows for c in row).lower()

    @property
    def header_row(self) -> Optional[list[str]]:
        """猜表头行：单位特征命中数最多的前 3 行。"""
        best: Optional[list[str]] = None
        best_hits = 0
        for row in self.rows[:3]:
            hits = sum(1 for c in row if _UNIT_CELL.search(_cell(c)))
            if hits > best_hits:
                best, best_hits = row, hits
        return best if best_hits else None

    @property
    def header_text(self) -> str:
        header = self.header_row
        return " | ".join(_cell(c) for c in header) if header else ""

    @property
    def data_row_count(self) -> int:
        """数据行数量：含 >= 3 个数值单元格的行。

        这是"真资源量表"与"正文被误判成表"的分水岭——正文碎片很难凑出
        连续多行的多个数值。定位时不靠它一个条件，但它是权重最高的一个。
        """
        count = 0
        for row in self.rows:
            numeric = 0
            for cell in row:
                text = _cell(cell)
                if not text:
                    continue
                if parse_number(text) is not None and any(ch.isdigit() for ch in text):
                    numeric += 1
            if numeric >= 3:
                count += 1
        return count

    def markdown(self) -> str:
        return render_table_markdown(self.rows)

    def score_against(self, page_text: str) -> float:
        """给表格打分：数据行数量 + 关键词 + 单位表头 + 同页上下文。"""
        flat = self.flat_text
        page_lower = page_text.lower()
        score = 3.0 * self.data_row_count
        score += 3.0 * sum(1 for t in _STRONG_TABLE_TERMS if t in flat)
        score += 1.0 * sum(1 for t in _MEDIUM_TABLE_TERMS if t in flat)
        score += 2.0 * sum(1 for t in ("indicated", "inferred") if t in page_lower)
        if self.header_row:
            score += 2.0
        # 数据行太少说明抽错了（比如抽到一句说明文字）
        if len(self.rows) < 2:
            score -= 4.0
        if all(len(r) < 3 for r in self.rows):
            score -= 2.0
        # 大量空行 = 正文被当成表
        if self.rows:
            empty_ratio = sum(1 for r in self.rows if not any(_cell(c) for c in r)) / len(self.rows)
            score -= 6.0 * empty_ratio
        return score


@dataclass
class PdfDocument:
    """一份 NI 43-101 PDF 的结构化视图。"""

    path: Path
    page_count: int = 0
    page_texts: list[str] = field(default_factory=list)
    tables: list[PdfTable] = field(default_factory=list)
    backend: str = "pdfplumber"
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def full_text(self) -> str:
        return "\n".join(self.page_texts)

    @property
    def text_length(self) -> int:
        return len(self.full_text.strip())

    @property
    def is_probably_scanned(self) -> bool:
        """扫描件（图片型 PDF）——本项目**未实现 OCR**，会显式报警而不是假装能读。

        判据用**平均每页**字符数而不是全文字符数：真实 NI 43-101 动辄上百页，
        用总量判断会把稀疏但正常的文档误判。
        """
        if self.page_count <= 0:
            return False
        return (self.text_length / self.page_count) < SCANNED_CHARS_PER_PAGE

    def candidate_tables(self, top_k: int = 4, min_score: float = 8.0) -> list[PdfTable]:
        """按得分返回最可能是资源量表的表格。

        ``min_score`` 默认 8.0：至少要凑出 2 个数据行（2 x 3.0）才有资格入围，
        这样可以挡掉"正文碎片被文本策略误判成表"的噪声。
        """
        ranked = sorted(self.tables, key=lambda t: t.score, reverse=True)
        picked = [t for t in ranked if t.score >= min_score and t.data_row_count >= 2]
        return picked[:top_k]

    def page_context(self, page_number: int, limit: int = 900) -> str:
        """取指定页的正文（脚注、cut-off grade 说明往往在这里）。"""
        if 1 <= page_number <= len(self.page_texts):
            return self.page_texts[page_number - 1][:limit]
        return ""

    # -- LLM 上下文 ------------------------------------------------------
    def extract_meta(self) -> dict[str, str]:
        """从全文中软提取文档元信息（company / project / date / cut-off）。

        这些只是**参考线索**，不是权威值；权威值由 Extractor 从表里抽。
        """
        text = self.full_text
        meta: dict[str, str] = {}

        # 先按"数字 + 单位"的形态抓，避免被 "0.50" 里的点截断
        m = re.search(
            r"cut-?off\s+grade\s+(?:of\s+)?"
            r"(\d+(?:\.\d+)?\s*(?:g/t|gpt|%|ppm)\s*[A-Za-z0-9]*)",
            text,
            re.IGNORECASE,
        ) or re.search(
            r"cut-?off\s+grade\s+(?:of\s+)?([^.;\n]{1,40})", text, re.IGNORECASE
        )
        if m:
            meta["cut_off_grade"] = m.group(1).strip()

        m = re.search(
            r"effective\s+date[^.\n]{0,40}?([A-Z][a-z]+\s+\d{1,2},\s*\d{4})",
            text,
            re.IGNORECASE,
        ) or re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        if m:
            meta["reported_date"] = m.group(1).strip()

        m = re.search(
            r"([A-Z][A-Za-z0-9&.,'\- ]{2,60}?(?:Corporation|Corp\.?|Limited|Ltd\.?|Inc\.?|"
            r"Minerals|Mining|Resources|Metals))",
            text,
        )
        if m:
            meta["company"] = m.group(1).strip()

        # 项目名容易误命中报告抬头（"...Technical Report - Mine"），
        # 这里排除掉含模板词的匹配。这是软线索，宁可缺也不要错。
        for match in re.finditer(
            r"([A-Z][A-Za-z0-9&.,'\- ]{2,60}?(?:Project|Mine|Deposit|Property|Complex))",
            text,
        ):
            candidate = match.group(1).strip()
            if any(bad in candidate for bad in ("Report", "Technical", "NI 43", "Statement")):
                continue
            meta["project"] = candidate
            break

        return meta

    def context_for_llm(self, max_chars: int = 14000, top_k: int = 3) -> str:
        """拼装给 LLM 的上下文：元信息 + 候选表 + 相关页正文片段。"""
        blocks: list[str] = []
        meta = self.extract_meta()
        if meta:
            lines = [f"- {k}: {v}" for k, v in meta.items()]
            blocks.append("## 文档元信息（自动识别，仅供参考）\n" + "\n".join(lines))

        chosen = self.candidate_tables(top_k=top_k)
        if chosen:
            parts = []
            for t in chosen:
                parts.append(
                    f"### 表（第 {t.page_number} 页，第 {t.index_on_page + 1} 张，"
                    f"定位得分 {t.score:.1f}）\n{t.markdown()}"
                )
            blocks.append("## 候选资源量表\n" + "\n\n".join(parts))
        else:
            blocks.append(
                "## 候选资源量表\n（未通过结构识别到表格，请从下方正文中提取）"
            )

        if chosen:
            seen_pages = sorted({t.page_number for t in chosen})
            snippets = []
            for pno in seen_pages[:3]:
                text = self.page_context(pno, limit=1800).strip()
                if text:
                    snippets.append(f"### 第 {pno} 页正文\n{text}")
            if snippets:
                blocks.append("## 相关页正文（含脚注）\n" + "\n\n".join(snippets))

        if not chosen:
            fallback = self.full_text[:max_chars]
            blocks.append("## 全文片段\n" + fallback)

        context = "\n\n".join(blocks)
        if len(context) > max_chars:
            context = context[:max_chars] + "\n\n...（上下文已截断）"
        return context


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------
def _page_texts_with_pymupdf(path: Path) -> list[str]:
    """pdfplumber 读不到文本时的兜底（例如字体嵌入异常）。"""
    try:
        import fitz  # PyMuPDF
    except ImportError:  # pragma: no cover
        return []
    try:
        with fitz.open(path) as doc:
            return [page.get_text() or "" for page in doc]
    except Exception:  # pragma: no cover
        return []


def load_pdf(path: Path, verbose: bool = False) -> PdfDocument:
    """读取 PDF：逐页取文本，并在含关键词的页上做表格结构识别。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 不存在：{path}")

    import pdfplumber

    doc = PdfDocument(path=path)
    page_texts: list[str] = []
    tables: list[PdfTable] = []

    with pdfplumber.open(str(path)) as pdf:
        doc.page_count = len(pdf.pages)
        for page_index, page in enumerate(pdf.pages):
            page_number = page_index + 1
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # pragma: no cover
                text = ""
                doc.warnings.append(f"第 {page_number} 页文本抽取失败：{exc}")
            page_texts.append(text)

            lowered = text.lower()
            if not any(k in lowered for k in PAGE_KEYWORDS):
                continue

            try:
                raw_tables = page.extract_tables()
            except Exception as exc:  # pragma: no cover
                doc.warnings.append(f"第 {page_number} 页表格抽取失败：{exc}")
                raw_tables = []

            if not raw_tables:
                # 无线框表（纯文本排版）兜底
                try:
                    raw_tables = page.extract_tables(
                        {
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                        }
                    )
                except Exception:  # pragma: no cover
                    raw_tables = []

            for table_index, raw in enumerate(raw_tables or []):
                rows = [[_cell(c) for c in row] for row in (raw or []) if row]
                if not rows:
                    continue
                table = PdfTable(
                    page_number=page_number, index_on_page=table_index, rows=rows
                )
                table.score = table.score_against(text)
                tables.append(table)

    if not any(t.strip() for t in page_texts):
        fallback = _page_texts_with_pymupdf(path)
        if fallback:
            page_texts = fallback
            doc.backend = "pymupdf"
            doc.warnings.append("pdfplumber 未取到文本，已切换 PyMuPDF 兜底。")

    doc.page_texts = page_texts
    doc.tables = sorted(tables, key=lambda t: (t.page_number, t.index_on_page))

    if doc.is_probably_scanned:
        doc.warnings.append(
            "疑似扫描件（可提取文本量过少）。本项目**未实现 OCR**，"
            "该文件将无法被正确抽取，会被标记为 abstain 而不是硬给结果。"
        )
    if verbose:
        for warning in doc.warnings:
            print(f"[pdf] {path.name}: {warning}")
    return doc


def discover_pdfs(pdf_dir: Path) -> list[Path]:
    """扫描目录下所有 PDF，按文件名排序（保证跑批顺序稳定可复现）。"""
    if not pdf_dir.exists():
        return []
    return sorted(p for p in pdf_dir.glob("*.pdf") if p.is_file())
