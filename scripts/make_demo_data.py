"""生成演示数据：3 份结构逼真的合成 NI 43-101 PDF + 对应的 ground truth JSON。

为什么需要它：题目给的真实 PDF 拿到之前，工程必须能端到端跑通并产出**真实可验证**
的结果。合成的表格里数字是自洽的（三角一致性成立），这样评测才有意义。

刻意埋了两类"考点"：

1. **合计行**（Newmont / Barrick 表尾的 ``Total`` 行）—— 不该被当成一个资源类别抽出来。
2. **原文自相矛盾的 Inferred 行**（Pilbara）—— 印刷的金属量 0.13 Mt 与
   ``32.7 Mt x 1.05% Li2O = 0.343 Mt`` 相差 62%，正确行为是**行级拒答**而不是硬给。
   这正是题目里"最看重"的那一项。

用法::

    python scripts/make_demo_data.py            # 写入 data/pdfs 与 data/ground_truth
    python scripts/make_demo_data.py --force    # 覆盖已存在的同名文件
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fitz  # PyMuPDF

DEFAULT_PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
DEFAULT_GT_DIR = PROJECT_ROOT / "data" / "ground_truth"

#: 行高必须留足——PyMuPDF 的 insert_textbox 一旦装不下就会**静默不渲染**，
#: 9pt 的 Helvetica 单行实际需要约 15pt 高度，这里给到 22pt。
ROW_HEIGHT = 22.0
CELL_FONT_SIZE = 9.0
BODY_FONT_SIZE = 9.5


def _fit_textbox(
    page: "fitz.Page",
    x0: float,
    y0: float,
    width: float,
    text: str,
    fontsize: float,
    fontname: str = "helv",
    color: tuple[float, float, float] = (0, 0, 0),
    max_height: float | None = None,
) -> float:
    """写入一段文本并返回实际占用高度。

    ``insert_textbox`` 返回值 < 0 表示装不下且**什么都不画**，所以这里逐步加高重试。
    """
    step = fontsize * 1.8
    height = step
    limit = max_height if max_height is not None else fontsize * 1.8 * 40
    while height <= limit:
        leftover = page.insert_textbox(
            fitz.Rect(x0, y0, x0 + width, y0 + height),
            text,
            fontsize=fontsize,
            fontname=fontname,
            align=fitz.TEXT_ALIGN_LEFT,
            color=color,
        )
        if leftover >= 0:
            return height
        height += step
    raise RuntimeError(f"文本装不进给定高度（{text[:40]!r}）")


# ==========================================================================
# 数据集定义
# ==========================================================================
NEWMONT = {
    "stem": "newmont_ni43-101",
    "company": "Newmont Corporation",
    "project": "Boddington Gold Mine",
    "title": "NI 43-101 Technical Report - Mineral Resource Estimate",
    "effective_date": "June 30, 2025",
    "cut_off": "0.50 g/t Au",
    "location": "Western Australia, Australia",
    "intro": [
        "This Technical Report has been prepared in accordance with the requirements of "
        "National Instrument 43-101 Standards of Disclosure for Mineral Projects.",
        "The Mineral Resource estimate for the Boddington Gold Mine is reported above a "
        "cut-off grade of 0.50 g/t Au. Mineral Resources are not Mineral Reserves and do "
        "not have demonstrated economic viability.",
        "All figures are rounded to reflect the relative accuracy of the estimates. "
        "Totals may not sum exactly due to rounding.",
    ],
    "table_page": 1,
    "table_caption": "Table 14-1: Boddington Gold Mine Mineral Resource Statement",
    "columns": ["Resource Category", "Tonnes (Mt)", "Grade (g/t Au)", "Contained Au (koz)"],
    "col_widths": [130.0, 85.0, 100.0, 115.0],
    "rows": [
        ["Indicated", "45.8", "1.21", "1,782"],
        ["Inferred", "23.6", "0.94", "713"],
        ["Total", "69.4", "-", "2,495"],
    ],
    "footnotes": [
        "Notes: (1) Cut-off grade 0.50 g/t Au. (2) Contained Au is calculated as "
        "tonnage multiplied by grade and converted to thousands of troy ounces.",
        "(3) Total row is the arithmetic sum and is provided for convenience only.",
    ],
    "gt_rows": [
        {
            "resource_class": "indicated",
            "commodity": "Au",
            "tonnage_mt": 45.8,
            "grade": 1.21,
            "grade_unit": "g/t",
            "contained_metal": 1782,
            "contained_metal_unit": "koz",
            "source_page": 1,
            "expect_abstain": False,
        },
        {
            "resource_class": "inferred",
            "commodity": "Au",
            "tonnage_mt": 23.6,
            "grade": 0.94,
            "grade_unit": "g/t",
            "contained_metal": 713,
            "contained_metal_unit": "koz",
            "source_page": 1,
            "expect_abstain": False,
        },
    ],
}

BARRICK = {
    "stem": "barrick_ni43-101",
    "company": "Barrick Gold Corporation",
    "project": "Reko Diq Copper-Gold Project",
    "title": "NI 43-101 Technical Report - Mineral Resource Estimate",
    "effective_date": "March 31, 2025",
    "cut_off": "0.20 % Cu",
    "location": "Balochistan, Pakistan",
    "intro": [
        "This Technical Report has been prepared in accordance with National Instrument "
        "43-101 Standards of Disclosure for Mineral Projects.",
        "The Mineral Resource estimate is reported above a cut-off grade of 0.20 % Cu. "
        "Copper and gold are reported as co-products; each is estimated independently "
        "from the same block model.",
        "Mineral Resources that are not Mineral Reserves do not have demonstrated "
        "economic viability. Numbers are rounded and totals may not sum exactly.",
    ],
    "table_page": 2,
    "table_caption": "Table 14-2: Reko Diq Mineral Resource Statement (Indicated and Inferred)",
    "columns": [
        "Resource Category",
        "Tonnes (Mt)",
        "Cu (%)",
        "Au (g/t)",
        "Contained Cu (t)",
        "Contained Au (koz)",
    ],
    "col_widths": [110.0, 75.0, 60.0, 60.0, 105.0, 105.0],
    "rows": [
        ["Indicated", "320.5", "0.42", "0.55", "1,346,100", "5,668"],
        ["Inferred", "145.2", "0.38", "0.48", "551,760", "2,241"],
        ["Total", "465.7", "-", "-", "1,897,860", "7,909"],
    ],
    "footnotes": [
        "Notes: (1) Cut-off grade 0.20 % Cu. (2) Contained Cu reported in metric tonnes; "
        "contained Au reported in thousands of troy ounces.",
        "(3) Total row is the arithmetic sum and is provided for convenience only.",
    ],
    "gt_rows": [
        {
            "resource_class": "indicated",
            "commodity": "Cu",
            "tonnage_mt": 320.5,
            "grade": 0.42,
            "grade_unit": "%",
            "contained_metal": 1346100,
            "contained_metal_unit": "t",
            "source_page": 2,
            "expect_abstain": False,
        },
        {
            "resource_class": "indicated",
            "commodity": "Au",
            "tonnage_mt": 320.5,
            "grade": 0.55,
            "grade_unit": "g/t",
            "contained_metal": 5668,
            "contained_metal_unit": "koz",
            "source_page": 2,
            "expect_abstain": False,
        },
        {
            "resource_class": "inferred",
            "commodity": "Cu",
            "tonnage_mt": 145.2,
            "grade": 0.38,
            "grade_unit": "%",
            "contained_metal": 551760,
            "contained_metal_unit": "t",
            "source_page": 2,
            "expect_abstain": False,
        },
        {
            "resource_class": "inferred",
            "commodity": "Au",
            "tonnage_mt": 145.2,
            "grade": 0.48,
            "grade_unit": "g/t",
            "contained_metal": 2241,
            "contained_metal_unit": "koz",
            "source_page": 2,
            "expect_abstain": False,
        },
    ],
}

PILBARA = {
    "stem": "pilbara_minerals_ni43-101",
    "company": "Pilbara Minerals Limited",
    "project": "Pilgangoora Lithium-Tantalum Project",
    "title": "NI 43-101 Technical Report - Mineral Resource Estimate",
    "effective_date": "August 14, 2025",
    "cut_off": "0.50 % Li2O",
    "location": "Pilbara Region, Western Australia",
    "intro": [
        "This Technical Report has been prepared in accordance with National Instrument "
        "43-101 Standards of Disclosure for Mineral Projects.",
        "The Mineral Resource estimate for the Pilgangoora Lithium-Tantalum Project is "
        "reported above a cut-off grade of 0.50 % Li2O.",
        "Mineral Resources are not Mineral Reserves and do not have demonstrated "
        "economic viability. Figures are rounded to reflect estimation accuracy.",
    ],
    "table_page": 1,
    "table_caption": "Table 14-3: Pilgangoora Mineral Resource Statement",
    "columns": ["Resource Category", "Tonnes (Mt)", "Li2O (%)", "Contained Li2O (Mt)"],
    "col_widths": [130.0, 90.0, 85.0, 130.0],
    # 注意：Inferred 行的 Contained Li2O 印刷值 0.13 与 32.7 Mt x 1.05% = 0.343 Mt
    # 相差 62%，是刻意埋的"原文自相矛盾"，正确行为是行级拒答。
    "rows": [
        ["Indicated", "64.3", "1.18", "0.76"],
        ["Inferred", "32.7", "1.05", "0.13"],
        ["Total", "97.0", "-", "0.89"],
    ],
    "footnotes": [
        "Notes: (1) Cut-off grade 0.50 % Li2O. (2) Contained Li2O reported in million "
        "tonnes. (3) Total row is the arithmetic sum and is provided for convenience only.",
    ],
    "gt_rows": [
        {
            "resource_class": "indicated",
            "commodity": "Li2O",
            "tonnage_mt": 64.3,
            "grade": 1.18,
            "grade_unit": "%",
            "contained_metal": 0.76,
            "contained_metal_unit": "Mt",
            "source_page": 1,
            "expect_abstain": False,
        },
        {
            "resource_class": "inferred",
            "commodity": "Li2O",
            "tonnage_mt": 32.7,
            "grade": 1.05,
            "grade_unit": "%",
            "contained_metal": 0.13,
            "contained_metal_unit": "Mt",
            "source_page": 1,
            "expect_abstain": True,
            "note": (
                "原文印刷的 Contained Li2O 0.13 Mt 与 32.7 Mt x 1.05% = 0.343 Mt "
                "相差 62%，自相矛盾。正确行为是行级拒答，转人工审核，而不是硬给。"
            ),
        },
    ],
}

DATASETS: tuple[dict[str, Any], ...] = (NEWMONT, BARRICK, PILBARA)


# ==========================================================================
# PDF 绘制
# ==========================================================================
def _draw_grid(
    page: "fitz.Page",
    x0: float,
    y0: float,
    col_widths: Sequence[float],
    row_count: int,
) -> None:
    total_w = sum(col_widths)
    total_h = row_count * ROW_HEIGHT
    for index in range(row_count + 1):
        y = y0 + index * ROW_HEIGHT
        page.draw_line(
            fitz.Point(x0, y), fitz.Point(x0 + total_w, y), color=(0.45, 0.45, 0.45), width=0.5
        )
    x = x0
    page.draw_line(
        fitz.Point(x, y0), fitz.Point(x, y0 + total_h), color=(0.45, 0.45, 0.45), width=0.5
    )
    for width in col_widths:
        x += width
        page.draw_line(
            fitz.Point(x, y0), fitz.Point(x, y0 + total_h), color=(0.45, 0.45, 0.45), width=0.5
        )


def _fill_table(
    page: "fitz.Page",
    x0: float,
    y0: float,
    col_widths: Sequence[float],
    rows: Sequence[Sequence[str]],
    bold_first_row: bool = True,
) -> float:
    for row_index, row in enumerate(rows):
        x = x0
        for col_index, cell in enumerate(row):
            width = col_widths[col_index] if col_index < len(col_widths) else col_widths[-1]
            _fit_textbox(
                page,
                x + 4,
                y0 + row_index * ROW_HEIGHT + 2,
                width - 8,
                str(cell),
                fontsize=CELL_FONT_SIZE,
                fontname="hebo" if (bold_first_row and row_index == 0) else "helv",
                max_height=ROW_HEIGHT - 4,
            )
            x += width
    return y0 + len(rows) * ROW_HEIGHT


def build_pdf(spec: dict[str, Any], out_path: Path) -> Path:
    doc = fitz.open()
    table_page = int(spec["table_page"])
    while len(doc) < table_page:
        page = doc.new_page(width=612, height=792)
        if len(doc) < table_page:
            _write_body_page(page, spec)

    page = doc[table_page - 1]
    _write_table_page(page, spec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    doc.close()
    return out_path


def _header_block(page: "fitz.Page", spec: dict[str, Any], top: float = 60.0) -> float:
    page.insert_text(
        fitz.Point(56, top),
        spec["company"],
        fontsize=13,
        fontname="hebo",
        color=(0, 0, 0),
    )
    page.insert_text(
        fitz.Point(56, top + 18),
        spec["title"],
        fontsize=10.5,
        fontname="helv",
        color=(0.15, 0.15, 0.15),
    )
    page.insert_text(
        fitz.Point(56, top + 34),
        f"Project: {spec['project']}    Location: {spec['location']}",
        fontsize=9,
        fontname="helv",
        color=(0.25, 0.25, 0.25),
    )
    page.insert_text(
        fitz.Point(56, top + 48),
        f"Effective date: {spec['effective_date']}    " f"Cut-off grade: {spec['cut_off']}",
        fontsize=9,
        fontname="helv",
        color=(0.25, 0.25, 0.25),
    )
    page.draw_line(
        fitz.Point(56, top + 58), fitz.Point(556, top + 58), color=(0.6, 0.6, 0.6), width=0.5
    )
    return top + 74


def _write_body_page(page: "fitz.Page", spec: dict[str, Any]) -> None:
    y = _header_block(page, spec)
    for paragraph in spec["intro"]:
        used = _fit_textbox(page, 56, y, 500, paragraph, BODY_FONT_SIZE)
        y += used + 14


def _write_table_page(page: "fitz.Page", spec: dict[str, Any]) -> None:
    y = _header_block(page, spec)
    for paragraph in spec["intro"][:2]:
        used = _fit_textbox(page, 56, y, 500, paragraph, BODY_FONT_SIZE)
        y += used + 14

    y += 6
    page.insert_text(
        fitz.Point(56, y), spec["table_caption"], fontsize=10, fontname="hebo", color=(0, 0, 0)
    )
    y += 14

    col_widths = list(spec["col_widths"])
    rows = [list(spec["columns"])] + [list(r) for r in spec["rows"]]
    _draw_grid(page, 56, y, col_widths, len(rows))
    bottom = _fill_table(page, 56, y, col_widths, rows)

    y = bottom + 18
    for note in spec["footnotes"]:
        used = _fit_textbox(page, 56, y, 500, note, 8.0, color=(0.2, 0.2, 0.2))
        y += used + 6


# ==========================================================================
# ground truth
# ==========================================================================
def build_ground_truth(spec: dict[str, Any], out_path: Path) -> Path:
    payload = {
        "pdf": f"{spec['stem']}.pdf",
        "company": spec["company"],
        "project": spec["project"],
        "reported_date": spec["effective_date"],
        "cut_off_grade": spec["cut_off"],
        "rows": spec["gt_rows"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="生成演示用 NI 43-101 PDF 与 ground truth")
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--force", action="store_true", help="覆盖已存在的同名文件")
    args = parser.parse_args()

    for spec in DATASETS:
        pdf_path = args.pdf_dir / f"{spec['stem']}.pdf"
        gt_path = args.gt_dir / f"{spec['stem']}.json"

        if pdf_path.exists() and not args.force:
            print(f"[skip] {pdf_path} 已存在（如需覆盖请加 --force）")
        else:
            build_pdf(spec, pdf_path)
            print(f"[ok]   {pdf_path}")

        if gt_path.exists() and not args.force:
            print(f"[skip] {gt_path} 已存在（如需覆盖请加 --force）")
        else:
            build_ground_truth(spec, gt_path)
            print(f"[ok]   {gt_path}")

    print("\n演示数据生成完成。下一步：python run_pipeline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
