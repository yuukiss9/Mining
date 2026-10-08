#!/usr/bin/env python
"""PDF 体检：判断一份 NI 43-101 是否**需要 OCR**。

回答的是一个具体问题：题目给的 PDF，文本层够不够用？

判定四条：
  R1 文本层是否完整        —— 每页平均字符数（文档级）
  R2 是否有整页图片         —— 文本少 + 图片覆盖面积大
  R3 资源量候选表是否落在图片页上 —— 最致命的情形：文档有文本层，但表是截图
  R4 表格是否靠文本层就能抽出来 —— 用项目自己的 pdf.py 跑一遍候选表定位

用法：
    python scripts/pdf_health.py                  # 体检 data/pdfs/ 下全部
    python scripts/pdf_health.py <pdf> [<pdf> ...]  # 体检指定文件
    python scripts/pdf_health.py --json           # 输出 JSON（给流水线消费）
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import fitz  # noqa: E402  PyMuPDF

from ni101.pdf import SCANNED_CHARS_PER_PAGE, discover_pdfs, load_pdf  # noqa: E402

# --------------------------------------------------------------------------
# 阈值（集中在此，便于按真实数据回调）
# --------------------------------------------------------------------------

# 注意：文档级"是否扫描件"的阈值**刻意不在此定义**，而是复用
# ``ni101.pdf.SCANNED_CHARS_PER_PAGE``。两处各写一个数字，就会出现
# "体检说需要 OCR、流水线却照常解析"的自相矛盾。这里只保留体检特有的
# 页面级阈值。

#: 单页字符数低于此值 -> 该页"文本稀疏"（比文档级阈值宽松：
#: 一页正常正文通常 1500+ 字符，低于 120 基本可以认定该页没承载正文）
SPARSE_PAGE_CHARS = 120

#: 单页图片覆盖面积占比超过此值 -> 该页"以图为主"
IMAGE_DOMINANT_RATIO = 0.45

#: 候选表得分达到该值才算"文本层能抽出来"
CANDIDATE_MIN_SCORE = 8.0


@dataclass
class PageStat:
    page: int
    chars: int
    images: int
    image_ratio: float
    sparse: bool
    image_dominant: bool
    suspected_image_page: bool


@dataclass
class PdfHealth:
    pdf: str
    pages: int
    text_chars: int
    chars_per_page: float
    image_count: int
    image_pages: int
    candidate_tables: int
    best_candidate_score: float
    best_candidate_page: int | None
    table_on_image_page: bool
    verdict: str
    needs_ocr: bool
    reasons: list[str] = field(default_factory=list)
    page_stats: list[PageStat] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("page_stats", None)  # JSON 输出不必带逐页明细
        return data


def _page_image_ratio(page: "fitz.Page") -> tuple[int, float]:
    """返回 (图片数, 图片覆盖面积占页面比例)。

    只统计**贴图块**（block type == 1），不把矢量图/线条算进去——
    NI 43-101 的表格边框是矢量线，算进来会虚高。
    """
    blocks = page.get_text("dict").get("blocks", [])
    image_blocks = [b for b in blocks if b.get("type") == 1]
    if not image_blocks:
        return 0, 0.0

    page_area = abs(page.rect.width * page.rect.height) or 1.0
    covered = 0.0
    for block in image_blocks:
        x0, y0, x1, y1 = block.get("bbox", (0, 0, 0, 0))
        covered += max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
    return len(image_blocks), min(1.0, covered / page_area)


def inspect(pdf_path: Path) -> PdfHealth:
    path = Path(pdf_path)
    with fitz.open(path) as doc:
        pages = doc.page_count
        page_texts = [doc[i].get_text() or "" for i in range(pages)]
        stats: list[PageStat] = []
        total_images = 0
        image_pages = 0
        for i in range(pages):
            text = page_texts[i]
            chars = len(text.strip())
            images, ratio = _page_image_ratio(doc[i])
            total_images += images
            sparse = chars < SPARSE_PAGE_CHARS
            image_dominant = ratio >= IMAGE_DOMINANT_RATIO
            suspected = sparse and image_dominant
            if suspected:
                image_pages += 1
            stats.append(
                PageStat(
                    page=i + 1,
                    chars=chars,
                    images=images,
                    image_ratio=round(ratio, 3),
                    sparse=sparse,
                    image_dominant=image_dominant,
                    suspected_image_page=suspected,
                )
            )

    text_chars = sum(len(t.strip()) for t in page_texts)
    chars_per_page = text_chars / pages if pages else 0.0

    # 复用项目自身的候选表定位，看文本层能不能真抽出资源量表
    document = load_pdf(path)
    candidates = document.candidate_tables()
    best = candidates[0] if candidates else None

    # 候选表所在页是否落在疑似图片页
    table_on_image_page = False
    if best is not None:
        for stat in stats:
            if stat.page == best.page_number and stat.suspected_image_page:
                table_on_image_page = True
                break

    reasons: list[str] = []
    needs_ocr = False
    #: 需要 OCR 的范围是整份还是仅个别页 —— 决定下面要不要给出成本提示
    full_ocr = False

    # 是否存在"文字密集页"——区分「整份扫描件」与「仅表格页是图」的关键。
    # 只要有一页文字完整，就不该把整份文档判成扫描件：那会让主人按全量 OCR
    # 的成本去处理本来只需 OCR 一页的情况。
    dense_pages = [s for s in stats if s.chars >= SPARSE_PAGE_CHARS]

    if pages == 0:
        verdict = "空文档"
        reasons.append("页数为 0，无法解析。")
        return PdfHealth(
            pdf=path.name, pages=0, text_chars=0, chars_per_page=0.0, image_count=0,
            image_pages=0, candidate_tables=0, best_candidate_score=0.0,
            best_candidate_page=None, table_on_image_page=False, verdict=verdict,
            needs_ocr=True, reasons=reasons, page_stats=stats,
        )

    # 判定顺序刻意如此：**能不能抽出表**是第一判据，字符数只是辅助。
    # 反过来先看字符数，会把"正文完整、只有表是截图"的文档误诊为整份扫描件，
    # 让主人按整份 OCR 的成本去处理本来只需 OCR 一页的情况。
    if best is not None and not table_on_image_page:
        verdict = "不需要 OCR —— 文本层可直接抽取"
        reasons.append(
            f"每页平均 {chars_per_page:.1f} 字符，文本层完整；"
            f"已从文本层定位到 {len(candidates)} 个候选表，"
            f"最优在第 {best.page_number} 页（得分 {best.score:.1f}）。"
        )
    elif best is not None and table_on_image_page:
        needs_ocr = True
        verdict = "仅资源量表页需要 OCR —— 表落在图片页上"
        reasons.append(
            f"最优候选表（第 {best.page_number} 页，得分 {best.score:.1f}）所在页"
            f"同时被判为疑似图片页，说明文字与表格可能来自不同图层；"
            f"**只需 OCR 该页，不必整份重跑**。"
        )
    elif not dense_pages and total_images > 0:
        needs_ocr = True
        full_ocr = True
        verdict = "扫描件 —— 整份需要 OCR"
        reasons.append(
            f"全部 {pages} 页都文字稀疏（没有一页达到 {SPARSE_PAGE_CHARS} 字符，"
            f"每页平均 {chars_per_page:.1f} 字符），且内容以图片块为主"
            f"（{total_images} 个），无任何一页存在可用文本层 —— 属真扫描件。"
        )
    elif not dense_pages:
        needs_ocr = True
        full_ocr = True
        verdict = "文本层稀疏且未抽到表 —— 需人工确认或补 OCR"
        reasons.append(
            f"全部 {pages} 页文字都稀疏（没有一页达到 {SPARSE_PAGE_CHARS} 字符，"
            f"每页平均 {chars_per_page:.1f} 字符），未见图片块，也未抽到任何表格结构。"
            f"可能是解析异常或文件内容极少，建议打开原文人工确认。"
        )
    elif not document.tables:
        needs_ocr = True
        verdict = "仅资源量表需要 OCR —— 表以图片嵌入正文"
        reasons.append(
            f"文档有 {len(dense_pages)} 页文字完整（非扫描件），但一个表格结构都抽不到。"
            f"典型情形是资源量表被做成截图贴进正文 —— **只需 OCR 该表所在页**，"
            f"正文页无需处理。"
        )
    else:
        needs_ocr = True
        verdict = "有文本层但抽不出资源量表 —— 疑似表格布局异常"
        reasons.append(
            f"共抽到 {len(document.tables)} 个表格结构，但没有一个够格当资源量表"
            f"（最高分 {max((t.score for t in document.tables), default=0.0):.1f} < "
            f"{CANDIDATE_MIN_SCORE}）。可能是跨页表 / 无表格线 / 文本框拼贴，"
            f"需要坐标聚类或视觉模型兜底。"
        )

    if image_pages and not needs_ocr:
        reasons.append(
            f"提示：有 {image_pages} 页以图为主，但这些页不承载资源量表，"
            f"不影响主流程（可能是封面、位置图、剖面图）。"
        )

    # 与流水线判据对齐：体检说"整份"时，run_pipeline 是否也会拒答？
    # 两个阈值同源（都取 ni101.pdf.SCANNED_CHARS_PER_PAGE），但体检的
    # "稀疏页"判据更敏感，所以字/页落在两者之间时会出现差异 —— 必须明说。
    if full_ocr:
        if chars_per_page < SCANNED_CHARS_PER_PAGE:
            reasons.append(
                f"与流水线一致：每页平均 {chars_per_page:.1f} < "
                f"{SCANNED_CHARS_PER_PAGE}（`ni101/pdf.py` 的 SCANNED_CHARS_PER_PAGE），"
                f"`run_pipeline.py` 会判为 `scanned_pdf` 并拒答，行为对得上。"
            )
        else:
            reasons.append(
                f"[!] 流水线尚未拒答：每页平均 {chars_per_page:.1f} 已不低于 "
                f"{SCANNED_CHARS_PER_PAGE}（`ni101/pdf.py` 的 SCANNED_CHARS_PER_PAGE），"
                f"`run_pipeline.py` 会照常解析但很可能抽不到内容。"
                f"若要统一行为，上调该常量即可（改一处，体检与流水线同时生效）。"
            )

    return PdfHealth(
        pdf=path.name,
        pages=pages,
        text_chars=text_chars,
        chars_per_page=round(chars_per_page, 1),
        image_count=total_images,
        image_pages=image_pages,
        candidate_tables=len(candidates),
        best_candidate_score=round(best.score, 1) if best else 0.0,
        best_candidate_page=best.page_number if best else None,
        table_on_image_page=table_on_image_page,
        verdict=verdict,
        needs_ocr=needs_ocr,
        reasons=reasons,
        page_stats=stats,
    )


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------

def _print_report(results: list[PdfHealth]) -> None:
    print("=" * 78)
    print("PDF 体检报告 —— 判断是否需要 OCR")
    print("=" * 78)

    header = f"{'文件':<34}{'页':>4}{'字/页':>9}{'图':>5}{'图片页':>7}{'候选表':>7}{'需OCR':>7}"
    print(header)
    print("-" * 78)
    for item in results:
        name = item.pdf if len(item.pdf) <= 32 else item.pdf[:29] + "..."
        print(
            f"{name:<34}{item.pages:>4}{item.chars_per_page:>9.1f}"
            f"{item.image_count:>5}{item.image_pages:>7}{item.candidate_tables:>7}"
            f"{'  是' if item.needs_ocr else '  否':>7}"
        )
    print("-" * 78)

    for item in results:
        print()
        print(f"### {item.pdf}")
        print(f"结论：{item.verdict}")
        for reason in item.reasons:
            print(f"  - {reason}")
        suspected = [s for s in item.page_stats if s.suspected_image_page]
        if suspected:
            pages = ", ".join(f"p{s.page}({s.image_ratio:.0%})" for s in suspected[:10])
            more = f" ...共 {len(suspected)} 页" if len(suspected) > 10 else ""
            print(f"  疑似图片页：{pages}{more}")

    total = len(results)
    need = sum(1 for r in results if r.needs_ocr)
    print()
    print("=" * 78)
    print(f"合计 {total} 份，其中 {need} 份需要 OCR，{total - need} 份不需要。")
    if need == 0:
        print("结论：当前数据集**不需要引入 OCR**，可直接用现有文本层链路。")
    else:
        print("结论：当前数据集**需要补 OCR**（或视觉模型）才能覆盖，见上方逐份原因。")
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="判断 NI 43-101 PDF 是否需要 OCR")
    parser.add_argument("paths", nargs="*", help="PDF 路径；省略则体检 data/pdfs/")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    if args.paths:
        pdfs = [Path(p) for p in args.paths]
    else:
        pdfs = discover_pdfs(PROJECT_ROOT / "data" / "pdfs")

    if not pdfs:
        print("没找到 PDF。请把文件放进 data/pdfs/，或显式传路径。")
        return 2

    results: list[PdfHealth] = []
    for pdf in pdfs:
        if not pdf.exists():
            print(f"[跳过] 不存在：{pdf}")
            continue
        try:
            results.append(inspect(pdf))
        except Exception as exc:  # noqa: BLE001 - 体检工具不该因单份文件崩掉
            print(f"[失败] {pdf.name}: {type(exc).__name__}: {exc}")

    if args.json:
        print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
    else:
        _print_report(results)

    return 0 if not any(r.needs_ocr for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
