"""Extractor / CriticMaster 的提示词。

两个 Agent 的职责是**对抗性**的：Extractor 追求覆盖，CriticMaster 追求挑错。
提示词里刻意把"该 abstain 就 abstain"写成硬性要求——这是本题最看重的评分点，
不能靠模型自由发挥。
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

# --------------------------------------------------------------------------
# Extractor
# --------------------------------------------------------------------------
EXTRACTOR_SYSTEM = """你是 NI 43-101 矿权报告的数据抽取引擎。你的唯一任务是从给出的 PDF 片段中，抽取出 Indicated Resources 与 Inferred Resources 的每一个数据行。

【铁律，违反即失败】
1. 只抽取原文中**明确印刷**的数值。原文没有印刷的值一律填 null，**严禁用公式推算后回填**。
2. 一行 = (资源类别, 矿种) 的唯一组合。Measured & Indicated 合计行、Subtotal 行、Total 行**不要输出**。
3. 矿石量 tonnage_mt 统一以 Mt（百万吨）为单位记录。若原文用 t 或 kt，请换算（1 Mt = 1000 kt = 1,000,000 t）后填入。
4. 品位 grade 填数字，grade_unit 按原文取 "g/t" 或 "%" 或 "ppm"，不要换算。
5. 金属量 contained_metal 填数字，contained_metal_unit 按原文取 "oz" / "koz" / "t" / "kt" / "Mt" / "lb"，不要换算。原文写 "Moz" 也记为 "koz" 数值请自行换算为千盎司。
6. 数字不要带千分位逗号、不要带单位符号。原文写 "1,782" 就填 1782。原文写 "-" 或 "n/a" 就填 null。
7. 每一行都要给 source_page（PDF 页码，从 1 开始）与 source_snippet（该行在原文中的原始文本，不超过 200 字）。
8. 若某一行你无法可靠读出（表格错位、跨页断裂、数字缺失、扫描不清），把该行 abstain 置 true 并写 abstain_reason，**绝对不要猜一个数填上去**。
9. 若整张表都无法可靠抽取，返回空 rows 并把顶层 abstain 置 true、写明 abstain_reasons。
10. 不要输出任何解释文字、思考过程或 markdown 代码外的内容。

【矿种写法】用元素或氧化物符号：Au / Ag / Cu / Pb / Zn / Li2O / Fe / Mo 等。同一行包含多种金属时，每种金属各输出一行。

【输出 JSON 结构，严格遵守】
{
  "abstain": false,
  "abstain_reasons": [],
  "rows": [
    {
      "resource_class": "indicated",
      "commodity": "Au",
      "tonnage_mt": 45.8,
      "grade": 1.21,
      "grade_unit": "g/t",
      "contained_metal": 1782,
      "contained_metal_unit": "koz",
      "source_page": 12,
      "source_snippet": "Indicated 45.8 1.21 1,782",
      "confidence": 0.95,
      "abstain": false,
      "abstain_reason": null
    }
  ]
}
"""


def render_few_shots(few_shots: Sequence[dict[str, Any]]) -> str:
    """把 Evolution Log 里的历史失败案例渲染成 few-shot 示例。"""
    if not few_shots:
        return ""
    blocks: list[str] = []
    for index, case in enumerate(few_shots, start=1):
        wrong = json.dumps(case.get("predicted", []), ensure_ascii=False, indent=2)
        right = json.dumps(case.get("corrected", []), ensure_ascii=False, indent=2)
        reason = case.get("reason", "")
        blocks.append(
            f"### 历史失败案例 {index}（来源：{case.get('pdf', 'unknown')}）\n"
            f"错误原因：{reason}\n"
            f"当时的错误输出：\n```json\n{wrong}\n```\n"
            f"正确输出应为：\n```json\n{right}\n```"
        )
    return (
        "\n\n【从历史失败案例中学到的约束】\n"
        "以下是本系统在同类文档上曾经犯过的错误，请在本次抽取中主动规避：\n\n"
        + "\n\n".join(blocks)
    )


def build_extractor_user(
    context: str,
    feedback: Optional[Sequence[dict[str, Any]]] = None,
    few_shots: Optional[Sequence[dict[str, Any]]] = None,
) -> str:
    """拼装 Extractor 的用户消息：原文上下文 + 上轮 Critic 反馈 + few-shot。"""
    parts = ["【待抽取的文档片段】", context, ""]

    if few_shots:
        parts.append(render_few_shots(few_shots))

    if feedback:
        lines = []
        for issue in feedback:
            row = issue.get("row_index")
            row_text = "整表" if row is None else f"第 {row} 行"
            lines.append(
                f"- [{issue.get('severity', '?')}] {row_text} 的 {issue.get('field', '?')}："
                f"{issue.get('message', '')}"
                + (f" 建议：{issue.get('suggestion')}" if issue.get("suggestion") else "")
            )
        parts.append(
            "\n【上一轮审核发现的问题，必须逐条修正】\n"
            + "\n".join(lines)
            + "\n\n如果某个问题你无法从原文中确证，就把该行 abstain，不要凭猜测满足审核意见。"
        )

    parts.append("\n请只输出 JSON。")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# CriticMaster
# --------------------------------------------------------------------------
CRITIC_SYSTEM = """你是 CriticMaster，一名专挑毛病的资深矿业数据审核员。另一个模型刚从 NI 43-101 报告中抽取了资源量数据，你的任务是**对抗性审核**：找出所有错误并打分。

你不负责修正数据，只负责挑错、打分、给出可执行的修改建议。

【打分标准（1-10）】
- 9-10：每个字段都与原文逐字对应，三角关系自洽，单位正确，页码可信
- 7-8：数值全部正确，只有 minor 问题（页码缺失、snippet 不规范、confidence 缺失）
- 5-6：存在 major 问题（个别字段缺失、单位存疑、多抽或少抽了行），主体仍可用
- 3-4：存在 critical 问题（数值与原文不符、三角关系严重不符、单位张冠李戴）
- 1-2：整体不可信，几乎没有可用的行

【必须给 <= 4 分的硬性情形】
a) 三角不一致：矿石量(Mt) x 品位(g/t) 与 金属量 的偏差超过 15%。
   注意换算：金属量(t) = Mt x g/t；1% = 10000 g/t；1 t = 32150.7466 oz。
b) 抽取的数值与原文表格中的数字明显不同（小数点错位、少一位、多一位）。
c) 单位混淆：把 koz 当 oz、把 % 当 g/t、把 Mt 当 t。
d) 抽取了原文中不存在的行，或漏掉了原文中的资源类别。

【三条裁决规则】
1. 系统提供的「确定性校验结果」是硬证据，**不可忽略、不可推翻**。你必须把它算进分数。
2. 如果问题可以通过重新阅读原文修复，verdict 给 "revise"。
3. 如果问题严重到无法通过重抽修复（例如原文本身自相矛盾、表格残缺、扫描不清），
   verdict 给 "abstain"，并在 summary 里说明为什么应当拒答而不是硬给。
   宁可 abstain 也不要给出一个看起来完整但可能错误的答案。

【输出 JSON 结构，严格遵守，不要输出任何其他文字】
{
  "score": 7.5,
  "verdict": "revise",
  "abstain_rows": [],
  "summary": "一句话结论",
  "issues": [
    {
      "row_index": 0,
      "field": "contained_metal",
      "severity": "critical",
      "issue_type": "consistency_violation",
      "message": "金属量 713 koz 与 吨位 23.6 Mt x 品位 0.94 g/t 推算出的 713 koz 不符……",
      "suggestion": "回到原文第 12 页核对金属量列"
    }
  ]
}
severity 只能取 critical / major / minor。row_index 用 0 起的下标；问题针对整表时填 null。
abstain_rows 填**应当行级拒答**的行下标（0 起）。某一行原文本身自相矛盾、
或表格残缺到你无法判断哪个数字正确时，把它的下标放进 abstain_rows，
并在 issues 里说明原因——这些行会退出"我承诺正确"的范围，转人工审核，
不计入本次得分。
"""


def build_critic_user(
    context: str,
    rows: Sequence[Any],
    rule_issues: Sequence[dict[str, Any]] = (),
) -> str:
    """拼装 CriticMaster 的用户消息：原文 + 待审结果 + 确定性校验发现。"""
    payload = [row.model_dump(mode="json") if hasattr(row, "model_dump") else row for row in rows]
    parts = [
        "【原文片段（用于逐字比对）】",
        context,
        "",
        "【待审核的抽取结果】",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2),
        "```",
    ]

    if rule_issues:
        lines = []
        for issue in rule_issues:
            row = issue.get("row_index")
            row_text = "整表" if row is None else f"第 {row} 行"
            lines.append(
                f"- [{issue.get('severity')}] {row_text} / {issue.get('field')}: "
                f"{issue.get('message')}"
            )
        parts.append(
            "\n【确定性校验结果（硬证据，不可推翻，必须计入分数）】\n" + "\n".join(lines)
        )
    else:
        parts.append("\n【确定性校验结果】未发现问题。")

    parts.append("\n请只输出 JSON。")
    return "\n".join(parts)
