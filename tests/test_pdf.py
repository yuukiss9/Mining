"""PDF 定位与表格打分的单测。

不依赖真实 PDF：直接构造 :class:`PdfTable` 来验证判据，
再单独用演示 PDF 验证端到端读取。
"""

from __future__ import annotations

from pathlib import Path

from ni101.pdf import PdfDocument, PdfTable, discover_pdfs, load_pdf, render_table_markdown

RESOURCE_TABLE = [
    ["Resource Category", "Tonnes (Mt)", "Grade (g/t Au)", "Contained Au (koz)"],
    ["Indicated", "45.8", "1.21", "1,782"],
    ["Inferred", "23.6", "0.94", "713"],
    ["Total", "69.4", "-", "2,495"],
]

PROSE_TABLE = [
    ["Mineral Resource", "estimate", ""],
    ["", "", ""],
    ["is reported above a", "cut-off grade", ""],
    ["", "", ""],
    ["Mineral Resources are not", "Mineral Reserves", ""],
]


class TestTableScoring:
    def test_real_resource_table_scores_high(self):
        table = PdfTable(page_number=1, index_on_page=0, rows=RESOURCE_TABLE)
        table.score = table.score_against("Indicated and Inferred Mineral Resources")
        assert table.data_row_count == 2
        assert table.score >= 8.0
        assert table.header_row is not None
        assert "Tonnes" in table.header_text

    def test_prose_is_rejected_by_data_row_gate(self):
        """正文碎片能蹭到关键词分，真正的闸门是"数据行数量"。"""
        table = PdfTable(page_number=1, index_on_page=0, rows=PROSE_TABLE)
        table.score = table.score_against("Indicated and Inferred Mineral Resources")
        assert table.data_row_count == 0

        doc = PdfDocument(
            path=Path("fake.pdf"), page_count=1,
            page_texts=["Indicated and Inferred Mineral Resources"],
        )
        doc.tables = [table]
        assert doc.candidate_tables() == []

    def test_candidate_filter_rejects_prose(self):
        doc = PdfDocument(path=Path("fake.pdf"), page_count=1, page_texts=["Indicated Inferred"])
        prose = PdfTable(page_number=1, index_on_page=0, rows=PROSE_TABLE)
        prose.score = prose.score_against("Indicated and Inferred Mineral Resources")
        real = PdfTable(page_number=1, index_on_page=1, rows=RESOURCE_TABLE)
        real.score = real.score_against("Indicated and Inferred Mineral Resources")
        doc.tables = [prose, real]

        picked = doc.candidate_tables()
        assert len(picked) == 1
        assert picked[0].rows[1][0] == "Indicated"

    def test_empty_row_ratio_penalty(self):
        rows = [["Indicated", "45.8", "1.21", "1782"], ["", "", "", ""], ["", "", "", ""]]
        table = PdfTable(page_number=1, index_on_page=0, rows=rows)
        table.score = table.score_against("Indicated")
        assert table.score < 3.0 * table.data_row_count + 8.0

    def test_bare_t_is_not_a_unit(self):
        """裸字母 t 不能当单位，否则 'category'/'estimate' 会被当成单位表头。"""
        noise = PdfTable(
            page_number=1, index_on_page=0,
            rows=[["Resource Category", "estimate", "cut-off grade"]],
        )
        assert noise.header_row is None

        real = PdfTable(
            page_number=1, index_on_page=0,
            rows=[["Resource Category", "Tonnes (Mt)", "Grade (g/t Au)"]],
        )
        assert real.header_row is not None


class TestRenderHelpers:
    def test_markdown_has_one_line_per_row(self):
        rendered = render_table_markdown(RESOURCE_TABLE)
        assert rendered.count("\n") == len(RESOURCE_TABLE) - 1
        assert "| Indicated | 45.8 | 1.21 | 1,782 |" in rendered

    def test_handles_none_cells(self):
        rendered = render_table_markdown([[None, "a"], ["b", None]])
        assert rendered == "|  | a |\n| b |  |"


class TestRealPdf:
    def test_loads_demo_pdfs(self, demo_pdf_dir: Path):
        pdfs = discover_pdfs(demo_pdf_dir)
        assert len(pdfs) >= 3

        for path in pdfs:
            doc = load_pdf(path)
            assert doc.page_count >= 1
            assert not doc.is_probably_scanned, path.name
            candidates = doc.candidate_tables()
            assert candidates, f"{path.name} 未定位到资源量表"
            assert candidates[0].data_row_count >= 2

    def test_meta_extraction(self, demo_pdf_dir: Path):
        doc = load_pdf(demo_pdf_dir / "newmont_ni43-101.pdf")
        meta = doc.extract_meta()
        assert "0.50 g/t Au" in meta.get("cut_off_grade", "")
        assert "Newmont" in meta.get("company", "")

    def test_context_for_llm_contains_table_and_footnotes(self, demo_pdf_dir: Path):
        doc = load_pdf(demo_pdf_dir / "newmont_ni43-101.pdf")
        context = doc.context_for_llm()
        assert "候选资源量表" in context
        assert "Contained Au (koz)" in context
        assert "cut-off grade" in context.lower()

    def test_context_is_truncated(self, demo_pdf_dir: Path):
        doc = load_pdf(demo_pdf_dir / "newmont_ni43-101.pdf")
        assert len(doc.context_for_llm(max_chars=300)) <= 340
