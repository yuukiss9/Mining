"""``scripts/pdf_health.py`` 的回归测试。

钉住两件事：
1. 真扫描件（无文本层）必须被判为"整份需要 OCR"
2. 正常文本层 PDF 必须被判为"不需要 OCR"

同时钉住一个曾踩过的坑：``insert_textbox`` 在文字溢出时**完全不渲染**，
用它构造测试夹具时如果框子太小，会造出"空页"从而让用例结论失真。
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pdf_health import inspect  # noqa: E402
from ni101.pdf import SCANNED_CHARS_PER_PAGE  # noqa: E402

#: 一段长度可控的正文，用于构造"文字完整"的页面
PARA = (
    "NI 43-101 Technical Report. The Mineral Resource estimate for the deposit is "
    "reported above a cut-off grade of 0.5 g/t Au. "
)


def _text_page(doc: fitz.Document, repeats: int = 24) -> fitz.Page:
    """造一页文字完整的页面（框子留够高，避免溢出导致静默不渲染）。"""
    page = doc.new_page(width=595, height=842)
    leftover = page.insert_textbox(
        fitz.Rect(56, 70, 540, 780), PARA * repeats, fontsize=10
    )
    assert leftover >= 0, "测试夹具溢出：insert_textbox 溢出时会完全不渲染，必须加大框子"
    return page


def _render_pixmap(doc: fitz.Document) -> "fitz.Pixmap":
    """把文字页渲染成位图，用于伪造扫描件。"""
    src = fitz.open()
    page = src.new_page(width=595, height=842)
    page.insert_textbox(
        fitz.Rect(56, 80, 540, 780),
        "Mineral Resource Statement\n"
        "Indicated 45.8 Mt @ 1.21 g/t for 1,782 koz\n"
        "Inferred 12.4 Mt @ 0.95 g/t for 379 koz\n" * 22,
        fontsize=11,
    )
    return page.get_pixmap(dpi=150)


class TestScannedPdf:
    def test_image_only_pdf_needs_full_ocr(self, tmp_path: Path):
        doc = fitz.open()
        pix = _render_pixmap(doc)
        for _ in range(2):
            page = doc.new_page(width=595, height=842)
            page.insert_image(page.rect, pixmap=pix, keep_proportion=True)
        target = tmp_path / "scanned.pdf"
        doc.save(target)

        result = inspect(target)
        assert result.needs_ocr is True
        assert "扫描件" in result.verdict
        # 阈值必须与流水线同源，否则会出现"体检说要 OCR、流水线照常解析"
        assert result.chars_per_page < SCANNED_CHARS_PER_PAGE
        assert result.candidate_tables == 0

    def test_empty_pdf_is_flagged(self, tmp_path: Path):
        doc = fitz.open()
        doc.new_page(width=595, height=842)  # 白页
        target = tmp_path / "blank.pdf"
        doc.save(target)

        result = inspect(target)
        # 白页既无文字也无图，必须报警而不是静默返回"不需要 OCR"
        assert result.needs_ocr is True


class TestTextPdf:
    def test_text_layer_pdf_does_not_need_ocr(self, tmp_path: Path):
        doc = fitz.open()
        _text_page(doc, 24)
        target = tmp_path / "digital.pdf"
        doc.save(target)

        result = inspect(target)
        assert result.chars_per_page > 100
        # 有文本层就没有"无文本层"的拒答理由；是否抽得到表由提取器负责，
        # 体检脚本只回答"文本层能不能用"
        assert not any("无可用文本层" in r for r in result.reasons)

    def test_mixed_pdf_distinguishes_table_page_from_full_scan(self, tmp_path: Path):
        """正文完整 + 单独一页表是截图 —— 不能被误诊成整份扫描件。

        这个区分很实际：整份 OCR 与只 OCR 一页是数量级的成本差。
        """
        doc = fitz.open()
        _text_page(doc, 24)
        table_page = doc.new_page(width=595, height=842)
        table_page.insert_textbox(
            fitz.Rect(56, 70, 540, 110),
            "Table 14-1: Mineral Resource Statement.",
            fontsize=10,
        )
        pix = _render_pixmap(doc)
        table_page.insert_image(
            fitz.Rect(56, 130, 551, 780), pixmap=pix, keep_proportion=True
        )
        target = tmp_path / "mixed.pdf"
        doc.save(target)

        result = inspect(target)
        # 存在文字密集页 -> 不能判成"整份扫描件"
        assert "整份需要 OCR" not in result.verdict
        assert result.chars_per_page > 100


class TestPipelineAgreement:
    """体检结论必须与流水线判据一致 —— 两处阈值不同源是真实踩过的坑。"""

    def test_full_scan_agrees_with_pipeline_flag(self, tmp_path: Path):
        doc = fitz.open()
        pix = _render_pixmap(doc)
        for _ in range(2):
            page = doc.new_page(width=595, height=842)
            page.insert_image(page.rect, pixmap=pix, keep_proportion=True)
        target = tmp_path / "scanned.pdf"
        doc.save(target)

        from ni101.pdf import load_pdf

        result = inspect(target)
        document = load_pdf(target)
        # 体检判"整份需要 OCR" <-> 流水线 is_probably_scanned 为真
        assert result.needs_ocr is True
        assert document.is_probably_scanned is True

    def test_digital_pdf_agrees_with_pipeline_flag(self, tmp_path: Path):
        doc = fitz.open()
        _text_page(doc, 24)
        target = tmp_path / "digital.pdf"
        doc.save(target)

        from ni101.pdf import load_pdf

        result = inspect(target)
        document = load_pdf(target)
        assert document.is_probably_scanned is False
        # 文本层可用时，体检不该报"无可用文本层"的理由
        assert not any("无可用文本层" in r for r in result.reasons)


class TestReportedFields:
    def test_page_stats_cover_every_page(self, tmp_path: Path):
        doc = fitz.open()
        _text_page(doc)
        doc.new_page(width=595, height=842)
        target = tmp_path / "two_pages.pdf"
        doc.save(target)

        result = inspect(target)
        assert result.pages == 2
        assert [s.page for s in result.page_stats] == [1, 2]
        assert result.page_stats[1].chars == 0
        assert result.page_stats[1].sparse is True

    def test_json_payload_drops_page_details(self, tmp_path: Path):
        doc = fitz.open()
        _text_page(doc)
        target = tmp_path / "one.pdf"
        doc.save(target)

        payload = inspect(target).to_dict()
        assert "page_stats" not in payload
        assert set(payload) >= {"pdf", "pages", "verdict", "needs_ocr", "reasons"}
