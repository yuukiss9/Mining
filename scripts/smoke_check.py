"""端到端自检脚本 —— 把调试时临时拼的验证固化成可复用的检查。

每个检查都打印 ``[PASS]/[FAIL] + 证据值``，证据值必须能一眼看出对不对
（打 ``True`` 是没有信息量的）。末尾汇总未通过项数并用退出码表达结果。

不依赖 PYTHONPATH，可任意目录直接运行::

    python scripts/smoke_check.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except (AttributeError, ValueError):  # pragma: no cover
    pass

from ni101.config import ensure_directories, load_config  # noqa: E402
from ni101.critic import CriticMasterAgent  # noqa: E402
from ni101.evolution import build_few_shots, load_cases  # noqa: E402
from ni101.extractor import ExtractorAgent  # noqa: E402
from ni101.llm import build_llm  # noqa: E402
from ni101.matcher import evaluate_run, find_ground_truth, load_ground_truth  # noqa: E402
from ni101.metrics import summarize  # noqa: E402
from ni101.pdf import discover_pdfs, load_pdf  # noqa: E402
from ni101.pipeline import STATUS_PARTIAL, STATUS_PASS, run_document  # noqa: E402
from ni101.schema import (  # noqa: E402
    GradeUnit,
    MetalUnit,
    consistency_deviation,
    expected_metal_tonnes,
    from_tonnes,
    parse_number,
    to_gr_per_tonne,
    to_tonnes,
)

_FAILURES: list[str] = []


def check(name: str, ok: bool, evidence: str) -> bool:
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"{tag} {name}")
    print(f"       证据：{evidence}")
    if not ok:
        _FAILURES.append(name)
    return ok


def section(title: str) -> None:
    print()
    print(f"=== {title} " + "=" * max(0, 66 - len(title)))


def main() -> int:
    print("=" * 74)
    print("NI 43-101 抽取流水线 —— 端到端自检")
    print("=" * 74)

    config = load_config()
    ensure_directories(config)

    # ---------------------------------------------------------------- 1
    section("1. 配置层")
    check(
        "config.yaml 可加载且降级状态明确",
        True,
        f"extractor={config.extractor.describe()} / critic={config.critic.describe()}",
    )
    mode = "真模型" if (config.extractor.configured and config.critic.configured) else "规则降级"
    print(f"       -> 当前运行模式：{mode}")

    # ---------------------------------------------------------------- 2
    section("2. 演示数据")
    pdfs = discover_pdfs(config.pdf_dir)
    if not pdfs:
        check("演示数据存在", False, f"{config.pdf_dir} 下没有 PDF，请先跑 scripts/make_demo_data.py")
        return 1
    gt_files = [find_ground_truth(p, config.ground_truth_dir) for p in pdfs]
    check(
        "PDF 与 ground truth 成对存在",
        all(gt is not None for gt in gt_files),
        f"pdf={len(pdfs)} 份，匹配到 gt={sum(1 for g in gt_files if g)} 份",
    )

    # ---------------------------------------------------------------- 3
    section("3. PDF 读取与表格定位")
    documents = {}
    for path in pdfs:
        doc = load_pdf(path)
        documents[path.name] = doc
        candidates = doc.candidate_tables()
        check(
            f"{path.name} 文本层可读",
            doc.text_length > 200 and not doc.is_probably_scanned,
            f"页数={doc.page_count} 文本长度={doc.text_length} 疑似扫描件={doc.is_probably_scanned}",
        )
        check(
            f"{path.name} 定位到资源量表",
            len(candidates) >= 1,
            f"候选表={len(candidates)} 首选表数据行={candidates[0].data_row_count if candidates else 0} "
            f"首选表表头={candidates[0].header_text if candidates else '-'}",
        )

    # ---------------------------------------------------------------- 4
    section("4. 单位换算与三角一致性公式")
    check(
        "三角公式 Mt x (g/t) == 金属量(t)",
        abs(expected_metal_tonnes(1.0, 1.0) - 1.0) < 1e-12,
        f"expected_metal_tonnes(1 Mt, 1 g/t) = {expected_metal_tonnes(1.0, 1.0)}",
    )
    check(
        "1% == 10000 g/t",
        abs(to_gr_per_tonne(1.0, GradeUnit.PERCENT) - 10000.0) < 1e-9,
        f"to_gr_per_tonne(1 %, %) = {to_gr_per_tonne(1.0, GradeUnit.PERCENT)}",
    )
    check(
        "1 公吨 == 32150.7466 金衡盎司",
        abs(from_tonnes(1.0, MetalUnit.OZ) - 32150.7466) < 1e-3,
        f"from_tonnes(1 t, oz) = {from_tonnes(1.0, MetalUnit.OZ):.4f}",
    )
    round_trip = from_tonnes(to_tonnes(1782.0, MetalUnit.KOZ), MetalUnit.KOZ)
    check(
        "koz <-> t 换算往返无损",
        abs(round_trip - 1782.0) < 1e-6,
        f"1782 koz -> t -> koz = {round_trip:.9f}",
    )
    check(
        "千分位与空值解析",
        parse_number("1,346,100") == 1346100.0 and parse_number("-") is None,
        f"parse_number('1,346,100')={parse_number('1,346,100')} "
        f"parse_number('-')={parse_number('-')}",
    )
    dev_ok = consistency_deviation(64.3, 1.18, GradeUnit.PERCENT, 0.76, MetalUnit.MT)
    dev_bad = consistency_deviation(32.7, 1.05, GradeUnit.PERCENT, 0.13, MetalUnit.MT)
    check(
        "三角一致性判据能区分自洽行与矛盾行",
        dev_ok is not None and dev_ok < 0.05 and dev_bad is not None
        and dev_bad > config.pipeline.consistency_hard_fail,
        f"Pilbara Indicated 偏差={dev_ok:.2%}（应 <5%）；"
        f"Inferred 偏差={dev_bad:.2%}（应 >{config.pipeline.consistency_hard_fail:.0%}）",
    )

    # ---------------------------------------------------------------- 5
    section("5. 抽取 + 对抗审核 + Revise Loop")
    extractor = ExtractorAgent(build_llm(config.extractor, retries=0))
    critic = CriticMasterAgent(build_llm(config.critic, retries=0), config.pipeline)

    pairs = []
    runs = {}
    for path in pdfs:
        run = run_document(path, config, extractor, critic, document=documents[path.name])
        gt_path = find_ground_truth(path, config.ground_truth_dir)
        truth = load_ground_truth(gt_path) if gt_path else None
        evaluation = evaluate_run(run, truth, config.pipeline.value_tolerance)
        run.evaluation = evaluation.to_dict()
        pairs.append((run, evaluation))
        runs[path.name] = (run, evaluation, truth)

        classes = sorted({row.resource_class.value for row in run.final_rows if row.resource_class})
        check(
            f"{path.name} 抽取出资源量行",
            len(run.final_rows) > 0,
            f"行数={len(run.final_rows)} 类别={classes} 抽取源={run.extraction_source} "
            f"最终得分={run.final_score:.1f}",
        )

    pilbara = next(
        (name for name in runs if "pilbara" in name.lower()), None
    )
    if pilbara:
        run, evaluation, _truth = runs[pilbara]
        abstained = [row for row in run.final_rows if row.abstain]
        reasons = " | ".join((row.abstain_reason or "")[:70] for row in abstained)
        check(
            "CriticMaster 抓出自相矛盾的行并触发行级拒答",
            len(abstained) >= 1,
            f"拒答行数={len(abstained)} 原因={reasons or '-'}",
        )
        check(
            "拒答行的三角偏差超过硬阈值",
            any((row.deviation or 0) > config.pipeline.consistency_hard_fail for row in abstained),
            "偏差=" + ", ".join(f"{row.identity_key()}:{(row.deviation or 0):.1%}" for row in abstained),
        )
        check(
            "该文档状态为 partial（部分拒答、其余通过）",
            run.status == STATUS_PARTIAL,
            f"status={run.status} 得分={run.final_score:.1f} 说明={run.note}",
        )
        check(
            "该行没有被硬给（核心评分点）",
            not evaluation.hard_give,
            f"hard_give={evaluation.hard_give} correct_abstain={evaluation.correct_abstain}",
        )

    clean_docs = [name for name in runs if "pilbara" not in name.lower()]
    check(
        "自洽的三份表全部通过",
        all(runs[name][0].status == STATUS_PASS for name in clean_docs),
        "; ".join(
            f"{name}: status={runs[name][0].status} 准确率={runs[name][1].field_accuracy:.1%}"
            for name in clean_docs
        ) or "无",
    )

    # ---------------------------------------------------------------- 6
    section("6. 评分协议（字段级 accuracy，容差 ±5%）")
    summary = summarize(pairs, config)
    check(
        "字段准确率 >= 90%",
        summary.field_accuracy >= 0.90,
        f"field_accuracy={summary.field_accuracy:.1%} ({summary.field_hit}/{summary.field_total})",
    )
    check(
        "硬给数 = 0（应拒答的行没有被硬给）",
        summary.hard_give_total == 0,
        f"hard_give={summary.hard_give_total}/{summary.expected_abstain_total} "
        f"hard_give_rate={summary.hard_give_rate:.1%}",
    )
    check(
        "正确拒答率 = 100%",
        summary.expected_abstain_total > 0
        and summary.correct_abstain_total == summary.expected_abstain_total,
        f"correct_abstain={summary.correct_abstain_total}/{summary.expected_abstain_total}",
    )

    # ---------------------------------------------------------------- 7
    section("7. Evolution Log 与 few-shot 回放")
    cases = load_cases(config.evolution_log)
    check(
        "evolution.jsonl 可回读",
        config.evolution_log.exists(),
        f"路径={config.evolution_log} 案例数={len(cases)}",
    )
    synthetic = [
        {
            "pdf": "__synthetic__.pdf",
            "created_at": "2026-01-01T00:00:00+00:00",
            "reason": "自检用合成案例",
            "predicted": [{"commodity": "Au", "tonnage_mt": 45.8}],
            "corrected": [{"commodity": "Au", "tonnage_mt": 45.9}],
        }
    ]
    few_shots = build_few_shots(synthetic, limit=3)
    check(
        "build_few_shots 能产出可用示例",
        len(few_shots) == 1 and few_shots[0]["predicted"] and few_shots[0]["corrected"],
        f"示例数={len(few_shots)} 含 predicted/corrected="
        f"{bool(few_shots and few_shots[0]['predicted'] and few_shots[0]['corrected'])}",
    )
    with tempfile.TemporaryDirectory() as tmp:
        from ni101.evolution import append_case

        tmp_log = Path(tmp) / "evolution.jsonl"
        append_case(tmp_log, synthetic[0])
        reloaded = load_cases(tmp_log)
        check(
            "append/load 往返一致",
            len(reloaded) == 1 and reloaded[0]["reason"] == synthetic[0]["reason"],
            f"往返案例数={len(reloaded)} reason={reloaded[0]['reason'] if reloaded else '-'}",
        )

    # ---------------------------------------------------------------- 8
    section("8. 输出产物")
    expected_outputs = [
        config.output_dir / "summary.json",
        config.output_dir / "report.md",
    ]
    for output in expected_outputs:
        check(
            f"{output.name} 已生成",
            output.exists() and output.stat().st_size > 0,
            f"路径={output} 大小={output.stat().st_size if output.exists() else 0} 字节",
        )
    extractions = list(config.extraction_dir.glob("*.json"))
    check(
        "每份 PDF 都有单文档记录",
        len(extractions) >= len(pdfs),
        f"extractions={len(extractions)} 份 / PDF={len(pdfs)} 份",
    )
    if config.output_dir.joinpath("summary.json").exists():
        payload = json.loads((config.output_dir / "summary.json").read_text(encoding="utf-8"))
        check(
            "summary.json 结构完整",
            {"field_accuracy", "hard_give_rate", "per_doc"} <= set(payload),
            f"keys={sorted(payload)[:8]}",
        )

    # ---------------------------------------------------------------- 汇总
    print()
    print("=" * 74)
    if _FAILURES:
        print(f"未通过 {len(_FAILURES)} 项：")
        for name in _FAILURES:
            print(f"  - {name}")
        print("=" * 74)
        return len(_FAILURES)
    print("全部检查通过。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
