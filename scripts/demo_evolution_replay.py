"""Evolution Log 回放演示 —— 证明 few-shot 改进机制真的接通了。

题目要求：「每次失败/降级 case 自动 append 到 evolution.jsonl，题目最后复跑这个 log，
看你是否能用它做 few-shot 改进」。本脚本把这条链路拆成两半分别验证：

**第一半（真实数据）** 读 outputs/evolution.jsonl，统计有多少条可用于 few-shot。
只有 ``predicted`` 与 ``corrected`` 都非空的案例才可学习——也就是"错的输出 + 对的输出"的对照。

**第二半（机制验证）** 用一个离线的脚本模型跑一遍 Extractor，
把注入到 prompt 里的 few-shot 片段原样打印出来。这一步不需要联网，
证明"注入确实发生了"，而不是配置写错了却看不出来。

> 注意边界：未配置真实模型时走的是**确定性规则抽取**，它压根不用 prompt，
> 因此 few-shot 不会改变抽取结果。要看到"分数真的提升"，需要先在 config.yaml
> 填好 Extractor / Critic 两端的 key，再跑 ``run_pipeline.py --replay-evolution``。

用法::

    python scripts/demo_evolution_replay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ni101.config import LLMSpec, load_config  # noqa: E402
from ni101.evolution import build_few_shots, load_cases  # noqa: E402
from ni101.extractor import ExtractorAgent  # noqa: E402
from ni101.llm import BaseLLM, LLMResult  # noqa: E402
from ni101.pdf import discover_pdfs, load_pdf  # noqa: E402

#: 演示用"历史失败案例"——形态与 evolution.jsonl 里的记录完全一致
DEMO_CASES = [
    {
        "pdf": "newmont_ni43-101.pdf",
        "created_at": "2026-01-01T00:00:00+00:00",
        "reason": "把 koz 当成了 oz：原文印刷 1,782 koz，抽成了 1,782,000 oz（数量级对了但单位错）",
        "predicted": [
            {
                "resource_class": "indicated",
                "commodity": "Au",
                "tonnage_mt": 45.8,
                "grade": 1.21,
                "grade_unit": "g/t",
                "contained_metal": 1782000,
                "contained_metal_unit": "oz",
            }
        ],
        "corrected": [
            {
                "resource_class": "indicated",
                "commodity": "Au",
                "tonnage_mt": 45.8,
                "grade": 1.21,
                "grade_unit": "g/t",
                "contained_metal": 1782,
                "contained_metal_unit": "koz",
            }
        ],
    },
    {
        "pdf": "barrick_ni43-101.pdf",
        "created_at": "2026-01-02T00:00:00+00:00",
        "reason": "多矿种同表时漏抽了 Au 列，只输出了 Cu 一行",
        "predicted": [
            {
                "resource_class": "indicated",
                "commodity": "Cu",
                "tonnage_mt": 320.5,
                "grade": 0.42,
                "grade_unit": "%",
                "contained_metal": 1346100,
                "contained_metal_unit": "t",
            }
        ],
        "corrected": [
            {
                "resource_class": "indicated",
                "commodity": "Cu",
                "tonnage_mt": 320.5,
                "grade": 0.42,
                "grade_unit": "%",
                "contained_metal": 1346100,
                "contained_metal_unit": "t",
            },
            {
                "resource_class": "indicated",
                "commodity": "Au",
                "tonnage_mt": 320.5,
                "grade": 0.55,
                "grade_unit": "g/t",
                "contained_metal": 5668,
                "contained_metal_unit": "koz",
            },
        ],
    },
]


class CapturingLLM(BaseLLM):
    """离线脚本模型：只捕获 prompt，不联网。"""

    def __init__(self) -> None:
        super().__init__(
            LLMSpec(label="capture", api_key="offline", base_url="http://127.0.0.1:0/v1",
                    model="offline-capture"),
            retries=0,
        )
        self.prompts: list[str] = []

    @property
    def available(self) -> bool:
        return True

    def complete(self, system: str, user: str, **kwargs: object) -> LLMResult:
        self.prompts.append(user)
        return LLMResult(ok=True, text='{"abstain": true, "rows": []}', model="offline-capture")


def main() -> int:
    config = load_config()

    print("=" * 74)
    print("第一半：真实 evolution.jsonl 的可学习案例统计")
    print("=" * 74)
    print(f"日志路径：{config.evolution_log}")

    cases = load_cases(config.evolution_log)
    usable = build_few_shots(cases, limit=config.pipeline.few_shot_limit)
    print(f"总案例数      ：{len(cases)}")
    print(f"可用于 few-shot：{len(usable)}（需要 predicted 与 corrected 都非空）")

    if cases:
        from collections import Counter

        triggers = Counter(t for case in cases for t in case.get("triggers", []))
        print("触发条件分布  ：" + ", ".join(f"{k}={v}" for k, v in triggers.most_common()))
    else:
        print("（日志为空，先跑一次 python run_pipeline.py 积累案例）")

    if not usable:
        print()
        print("说明：当前案例都是「降级记录」（例如抽到了但走了规则路径），")
        print("      没有形成「错误输出 -> 正确输出」的对照，所以不可学习。")
        print("      要产出可学习案例，需要配置真实模型后跑出字段级错误或硬给。")

    print()
    print("=" * 74)
    print("第二半：few-shot 注入机制验证（离线，不联网）")
    print("=" * 74)

    pdfs = discover_pdfs(config.pdf_dir)
    if not pdfs:
        print("data/pdfs 下没有 PDF，先跑 scripts/make_demo_data.py")
        return 2
    document = load_pdf(pdfs[0])

    llm = CapturingLLM()
    agent = ExtractorAgent(llm)
    agent.extract(document, few_shots=DEMO_CASES)

    prompt = llm.prompts[0]
    marker = "【从历史失败案例中学到的约束】"
    if marker not in prompt:
        print("[FAIL] few-shot 没有注入到 Extractor 的 prompt")
        return 1

    print(f"[PASS] 已注入 {len(DEMO_CASES)} 条历史失败案例到 Extractor prompt")
    print(f"       目标文档：{document.name}")
    print(f"       prompt 总长度：{len(prompt)} 字符")
    print()
    print("---- 注入片段（原样摘自实际 prompt） ----")
    start = prompt.index(marker)
    print(prompt[start : start + 900])
    print("---- 片段结束 ----")
    print()

    if config.extractor.configured:
        print("检测到 Extractor 已配置真实模型。")
        print("执行 `python run_pipeline.py --replay-evolution` 即可看到 few-shot 对抽取结果的实际影响。")
    else:
        print("检测到 Extractor 未配置：运行时走确定性规则路径，不消费 prompt。")
        print("配置 config.yaml 的「填写点 1 / 2」后，同样的注入会真实影响抽取结果。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
