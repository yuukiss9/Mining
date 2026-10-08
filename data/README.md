# 数据目录说明

> **本仓库不携带数据文件** —— `.gitignore` 排除了 `data/pdfs/*.pdf` 与
> `data/ground_truth/*.json`，避免真实报告被误提交。
> 想立刻跑通全流程，先执行 `python scripts/make_demo_data.py` 生成合成演示数据。

## data/pdfs/ —— 放待抽取的 NI 43-101 原始 PDF

把待处理的 NI 43-101 PDF 丢进这个目录即可，文件名随意，
流水线会自动扫描该目录下所有 `*.pdf`。

`make_demo_data.py` 生成的合成演示数据是三份：

| 公司 | 矿种 | 备注 |
| --- | --- | --- |
| Newmont | Au（金） | 单矿种表 |
| Barrick | Au + Cu（铜金） | 双矿种并列，考验列配对 |
| Pilbara Minerals | Li2O（锂） | 含一行原文**自相矛盾**的金属量，用于验证行级拒答 |

> 拿到真实 PDF 后第一件事：跑 `python scripts/pdf_health.py` 体检文本层，
> 判断需不需要 OCR（见 README 第 9 节）。

## data/ground_truth/ —— 放 ground truth JSON

题目提供的 ground truth JSON 放这里。文件名与 PDF 的 stem 对应，例如：

```
data/pdfs/newmont_ni43-101.pdf   ->   data/ground_truth/newmont_ni43-101.json
data/pdfs/barrick_ni43-101.pdf   ->   data/ground_truth/barrick_ni43-101.json
```

### JSON 结构约定

```json
{
  "pdf": "newmont_ni43-101.pdf",
  "company": "Newmont Corporation",
  "project": "Boddington Gold Mine",
  "reported_date": "2025-06-30",
  "cut_off_grade": "0.5 g/t Au",
  "rows": [
    {
      "resource_class": "indicated",
      "commodity": "Au",
      "tonnage_mt": 45.8,
      "grade": 1.21,
      "grade_unit": "g/t",
      "contained_metal": 1782.0,
      "contained_metal_unit": "koz",
      "source_page": 1,
      "expect_abstain": false
    }
  ]
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `resource_class` | `measured` / `indicated` / `inferred` |
| `commodity` | 元素或氧化物符号，如 `Au`、`Cu`、`Ag`、`Li2O` |
| `tonnage_mt` | 矿石量，单位百万吨（Mt） |
| `grade` + `grade_unit` | 品位，单位 `g/t` 或 `%`（也接受 `ppm`） |
| `contained_metal` + `contained_metal_unit` | 金属量，单位 `oz` / `koz` / `t` / `kt` / `Mt` / `lb` |
| `source_page` | 该行在 PDF 中的页码（1-based），用于来源可追溯校验 |
| `expect_abstain` | **核心标记**。`true` 表示该行在原文中数值自相矛盾（吨位 x 品位 与金属量对不上），**正确行为是拒答而不是硬给** |

> `expect_abstain: true` 是本题评分协议里"最看重"的那一项：
> 抽取明显错误时系统是否 abstain。评测脚本会把它计入 `硬给率`。

### 若题目给的 ground truth 结构不同

本项目对 GT 的读取集中在 `ni101/matcher.py::load_ground_truth()`，
字段映射若不一致，只改这一个函数即可，其余代码不动。
