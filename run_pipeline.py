"""主入口：跑完整条 NI 43-101 抽取 + 对抗审核 + 评测流水线。

用法::

    python run_pipeline.py                      # 跑 data/pdfs 下全部 PDF
    python run_pipeline.py --replay-evolution   # 用 evolution.jsonl 做 few-shot 复跑
    python run_pipeline.py --only pilbara       # 只跑文件名含该子串的 PDF
    python run_pipeline.py --quiet              # 只输出汇总

产物（默认写到 outputs/）::

    outputs/extractions/<stem>.json   每份 PDF 的完整往返记录（含每轮 Critic 结果）
    outputs/summary.json              总览指标
    outputs/report.md                 可直接粘贴进汇报的 markdown 报告
    outputs/evolution.jsonl           失败/降级案例（追加式），可回放做 few-shot

退出码：0 = 没有"硬给"；1 = 存在硬给（应拒答却给了数值）。可直接当 CI 闸门用。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 子进程/重定向时避免父进程 print 被缓冲到子进程输出之后
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except (AttributeError, ValueError):  # pragma: no cover
    pass

from ni101.config import ensure_directories, load_config  # noqa: E402
from ni101.critic import CriticMasterAgent  # noqa: E402
from ni101.evolution import append_case, build_few_shots, load_cases, make_case, should_log  # noqa: E402
from ni101.extractor import ExtractorAgent  # noqa: E402
from ni101.llm import build_llm  # noqa: E402
from ni101.matcher import evaluate_run, find_ground_truth, load_ground_truth  # noqa: E402
from ni101.metrics import render_markdown_report, summarize  # noqa: E402
from ni101.pdf import discover_pdfs  # noqa: E402
from ni101.pipeline import run_document  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NI 43-101 资源量抽取 + 对抗审核流水线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--pdf-dir", type=Path, default=None, help="默认取 config.yaml 的 paths.pdf_dir")
    parser.add_argument("--gt-dir", type=Path, default=None, help="默认取 config.yaml 的 paths.ground_truth_dir")
    parser.add_argument("--out", type=Path, default=None, help="默认取 config.yaml 的 paths.output_dir")
    parser.add_argument("--only", type=str, default=None, help="只跑文件名含该子串的 PDF")
    parser.add_argument("--max-rounds", type=int, default=None, help="覆盖 Revise Loop 最大轮次")
    parser.add_argument("--pass-score", type=float, default=None, help="覆盖 CriticMaster 通过分")
    parser.add_argument(
        "--replay-evolution",
        action="store_true",
        help="读取 evolution.jsonl 中的历史失败案例，作为 few-shot 注入 Extractor",
    )
    parser.add_argument("--quiet", action="store_true", help="只输出最终汇总")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    if args.pdf_dir:
        config.paths.pdf_dir = str(args.pdf_dir)
    if args.gt_dir:
        config.paths.ground_truth_dir = str(args.gt_dir)
    if args.out:
        config.paths.output_dir = str(args.out)
    if args.max_rounds is not None:
        config.pipeline.max_rounds = args.max_rounds
    if args.pass_score is not None:
        config.pipeline.pass_score = args.pass_score

    ensure_directories(config)

    log = (lambda *_: None) if args.quiet else (lambda msg: print(msg))

    print("=" * 78)
    print("NI 43-101 资源量抽取 + 对抗审核流水线")
    print("=" * 78)
    print("[配置]")
    print(config.describe())
    print(f"[路径] pdf_dir={config.pdf_dir}")
    print(f"       gt_dir ={config.ground_truth_dir}")
    print(f"       out_dir={config.output_dir}")
    if not config.extractor.configured or not config.critic.configured:
        print(
            "[提示] 存在未配置的模型端点，对应环节将走**确定性规则**降级路径。\n"
            "       运行 `python scripts/check_llm.py` 可以确认配置是否真的生效。"
        )
    print()

    extractor = ExtractorAgent(
        build_llm(config.extractor, retries=config.runtime.request_retries),
        verbose=not args.quiet,
    )
    critic = CriticMasterAgent(
        build_llm(config.critic, retries=config.runtime.request_retries),
        config.pipeline,
        verbose=not args.quiet,
    )

    few_shots: list[dict] = []
    if args.replay_evolution:
        cases = load_cases(config.evolution_log)
        few_shots = build_few_shots(cases, limit=config.pipeline.few_shot_limit)
        print(
            f"[Evolution] 载入 {len(cases)} 条历史案例，"
            f"其中 {len(few_shots)} 条可用于 few-shot 注入\n"
        )
        if not cases:
            print("[Evolution] 日志为空，本次与普通跑批等价（先跑一次积累失败案例）\n")

    pdf_paths = discover_pdfs(config.pdf_dir)
    if args.only:
        pdf_paths = [p for p in pdf_paths if args.only.lower() in p.name.lower()]
    if not pdf_paths:
        print(
            f"[错误] {config.pdf_dir} 下没有 PDF。\n"
            "       放入真实 NI 43-101 PDF，或先跑 `python scripts/make_demo_data.py` 生成演示数据。"
        )
        return 2

    print(f"[跑批] 共 {len(pdf_paths)} 份 PDF\n")

    pairs = []
    for pdf_path in pdf_paths:
        print(f"[{pdf_path.name}]")
        run = run_document(
            pdf_path,
            config,
            extractor,
            critic,
            few_shots=few_shots or None,
            log=log,
        )

        gt_path = find_ground_truth(pdf_path, config.ground_truth_dir)
        truth = load_ground_truth(gt_path) if gt_path else None
        evaluation = evaluate_run(run, truth, config.pipeline.value_tolerance)
        run.evaluation = evaluation.to_dict()
        pairs.append((run, evaluation))

        if truth is None:
            print(f"  (!) 未找到 ground truth（{config.ground_truth_dir}），跳过评测\n")
        else:
            print(
                f"  => 状态={run.status} 得分={run.final_score:.1f} "
                f"字段准确率={evaluation.field_accuracy:.1%} "
                f"硬给={len(evaluation.hard_give)} 正确拒答={len(evaluation.correct_abstain)}\n"
            )

        # 产出单文档记录
        (config.extraction_dir / f"{run.pdf_stem}.json").write_text(
            json.dumps(run.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 失败/降级案例自动落盘
        trigger, matched = should_log(run, evaluation, config)
        if trigger:
            case = make_case(
                run,
                evaluation,
                truth_rows=truth.rows if truth else (),
                triggers=matched,
            )
            append_case(config.evolution_log, case)
            print(f"  => 已记录到 evolution.jsonl（触发条件：{', '.join(matched)}）\n")

    summary = summarize(pairs, config)
    report = render_markdown_report(summary, pairs, config)

    (config.output_dir / "summary.json").write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (config.output_dir / "report.md").write_text(report, encoding="utf-8")

    print("=" * 78)
    print("汇总")
    print("=" * 78)
    print(
        f"文档数           : {summary.docs}  "
        f"(通过 {summary.status_counts.get('pass', 0)} / "
        f"部分通过 {summary.status_counts.get('partial', 0)} / "
        f"拒答 {summary.status_counts.get('abstain', 0)})"
    )
    print(f"字段级准确率     : {summary.field_accuracy:.1%}  ({summary.field_hit}/{summary.field_total})")
    print(f"行精确率/召回率  : {summary.row_precision:.1%} / {summary.row_recall:.1%}")
    print(
        f"硬给率           : {summary.hard_give_rate:.1%}  "
        f"({summary.hard_give_total}/{summary.expected_abstain_total})  <- 越低越好"
    )
    print(f"正确拒答率       : {summary.correct_abstain_rate:.1%}")
    print(f"硬给文档数       : {summary.wrong_but_confident_docs}")
    print()
    print(f"详细报告: {config.output_dir / 'report.md'}")
    print(f"总览指标: {config.output_dir / 'summary.json'}")
    print(f"进化日志: {config.evolution_log}")

    return 1 if summary.hard_give_total > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
