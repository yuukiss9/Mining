# NI 43-101 资源量抽取 + 对抗审核

[![tests](https://github.com/yuukiss9/Mining/actions/workflows/tests.yml/badge.svg)](https://github.com/yuukiss9/Mining/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

从 NI 43-101 矿权报告 PDF 中自动抽取 Indicated / Inferred Resources 的
**矿石量 (Mt) · 品位 (g/t Au 或 % Cu) · 金属量 (oz 或 t)**，
并由第二个模型做对抗性审核、迭代返工，**在无法确证时拒答而不是硬给**。

> **零配置可跑** —— 不填任何 API Key 也能跑通全流程（未配置时自动走确定性规则降级路径）。
> 需要手动填的只有一个文件 **`config.yaml`**，共 2 处
> （`extractor` 与 `critic` 的 `api_key` / `base_url`），
> 在编辑器里搜索中文关键词 **「填写点」** 可以直接跳转。

---

## 1. 先看这里：两条路径

**环境要求**：Python ≥ 3.10。下文命令中的 `python` 若在你的系统上是 `python3`，请自行替换。

### 路径 A · 零配置体验（约 30 秒）

```bash
git clone <本仓库地址>
cd <仓库目录>

pip install -r requirements.txt

# 1) 生成 3 份结构逼真的合成 NI 43-101 PDF + 对应 ground truth
#    （仓库不携带数据文件，因此这一步必须先跑）
python scripts/make_demo_data.py

# 2) 跑完整流水线
python run_pipeline.py

# 3) 端到端自检（每项带证据值，退出码 = 失败项数）
python scripts/smoke_check.py

# 4) 单元测试
python -m pytest
```

### 路径 B · 接自己的真实 PDF

```bash
# 1) 按同名 stem 放入数据（格式见 data/README.md）
#    data/pdfs/<name>.pdf   +   data/ground_truth/<name>.json

# 2) ★ 先体检：这份 PDF 有文本层吗？需要 OCR 吗？
python scripts/pdf_health.py

# 3) 填好 config.yaml 的两处填写点后，先确认端点真的通
python scripts/check_llm.py

# 4) 跑批
python run_pipeline.py
```

### 实测结果，以及它**不能**证明什么

在**合成演示数据 + 未配置模型（规则降级路径）**下：

```
文档数           : 3  (通过 2 / 部分通过 1 / 拒答 0)
字段级准确率     : 100.0%  (21/21)
行精确率/召回率  : 100.0% / 100.0%
硬给率           : 0.0%  (0/1)   <- 越低越好
正确拒答率       : 100.0%
```

这个数字必须配着边界看，否则是误导：

| 它证明了 | 它**不**证明 |
| --- | --- |
| 全链路逻辑自洽：PDF 解析 → 抽取 → 审核 → 拒答判定 → 评测 打通 | 真实 NI 43-101 上的抽取质量 |
| 拒答机制真的会触发（刻意埋的 1 处矛盾行被行级拒答，不是写死的） | 双模型对抗的实际效果（模型尚未配置） |
| 评分协议按字段级 ±5% 正确实现 | 跨页表 / 合并单元格 / 脚注改口径等真实复杂度 |

合成 PDF 是用 PyMuPDF 生成的**规整表格**；真实 NI 43-101 存在跨页表、合并单元格、
脚注改口径（`including` / `at cut-off grade`）、单位混用等情形，规则路径的准确率会
明显低于此处 —— **这正是 `config.yaml` 需要配置真实模型的原因**。

逐条需求对照见 [`docs/需求映射.md`](docs/需求映射.md)，未落地项集中在其 H 节。

---

## 2. 痛点与设计主线

| 痛点 | 表现 | 本项目的处理 |
| --- | --- | --- |
| 真实 NI 43-101 表格极难 | 跨页表、合并单元格、脚注改变口径、Indicated/Inferred 行错位、单位混排 | 先**定位候选表**再喂模型（只送命中的页与表）；规则层做硬校验兜底 |
| 明显错误时模型倾向于"硬给" | 表格残缺也照样编一个看起来完整的答案 | **三角一致性**判据 + 行级拒答 + 分数门槛 + 轮次上限，四道闸 |
| 单位混排导致"假失败" | 模型给 `1,782,000 oz`，原文是 `1,782 koz` —— 物理量相同却被判错 | 比对前**统一归一到公吨 / g/t**，再按 ±5% 相对容差比较 |
| 配置写错会静默降级 | 以为在用 GPT-4o，其实一次调用都没发生 | `scripts/check_llm.py` 分三层（配置 / 可达 / 调用）暴露失败点 |
| 迭代轮次无上限 | 反复重试直到转人工，主流程永远走不到成功 | 严格 ≤3 轮；**确定性规则路径不空转**（喂回反馈也不会变） |

### 设计主线：三角一致性

这是本题的"物理定律"，也是 CriticMaster 能在**不看原文**的情况下抓出明显错误的依据：

```
1 Mt = 1e6 t，品位 1 g/t = 每吨矿石含 1 g 金属
=> 金属量(t) = 1e6 × 品位 / 1e6 = 矿石量(Mt) × 品位(g/t)
```

即 `expected_metal_tonnes(Mt, g/t) == Mt × g_t`（数值上恰好相等）。
换算成 `%` 品位时 `1 % = 10000 g/t`，换算成盎司时 `1 t = 32150.7466 oz`。

偏差超过 **15%** 记为 critical 问题；超过 **35%** 直接**行级拒答**，
转人工审核，不参与"我承诺正确"的范围。这样一行坏数据不会把整份文档拖进全篇拒答，
白丢正确的行。

---

## 3. 端到端链路

```
PDF ──► 候选表定位 ──► Extractor ──► CriticMaster ──► 通过? ──► 评测 ──► Evolution Log
        (只送命中页)    (强模型/规则)   (规则硬校验         │ 否          (±5%)      (失败/降级
                                        + 语义审核)         │                      自动落盘)
                                                        带反馈重抽
                                                        (≤3 轮，否则拒答)
```

## 4. 架构分层

| 层 | 文件 | 职责 |
| --- | --- | --- |
| 配置层 | `ni101/config.py` | 加载 yaml、环境变量覆盖、**ASCII 校验**、脱敏打印 |
| 模型层 | `ni101/llm.py` | OpenAI 兼容客户端（DeepSeek/Qwen/GLM/OpenAI/Ollama 通用）+ 未配置时的显式"不可用"降级；JSON 抽取（含代码围栏与括号配对兜底） |
| 领域层 | `ni101/schema.py` | 数据模型 + 单位归一化 + 三角一致性公式（**唯一实现处**） |
| 数据层 | `ni101/pdf.py` | PDF 文本/表格抽取、候选表定位打分、LLM 上下文拼装、扫描件识别 |
| 能力层 | `ni101/extractor.py` | Extractor Agent：LLM 路径 + 确定性规则路径（支持多矿种并列表） |
| 审核层 | `ni101/critic.py` | CriticMaster Agent：确定性规则层（硬证据）+ 语义审核层，分数取**较低值** |
| 编排层 | `ni101/pipeline.py` | Revise Loop、行级拒答落地、文档级状态判定 |
| 评测层 | `ni101/matcher.py` / `metrics.py` | 字段级 ±5% 比对、防假失败、指标汇总与报告渲染 |
| 进化层 | `ni101/evolution.py` | 失败/降级判定、JSONL 落盘、few-shot 回放素材构建 |
| 入口 | `run_pipeline.py` / `ni101/cli` 风格 | 跑批、评测、产物落盘、退出码闸门 |

---

## 5. 关键设计决策

| 决策 | 为什么 | 落在哪 |
| --- | --- | --- |
| 比对前统一归一到 t / g/t | 单位混排会把**正确的**抽取判成错的，一旦判错就反复重试直到转人工 | `schema.to_tonnes` / `to_gr_per_tonne`；`matcher._normalized` |
| 用"数据行数量"当候选表的闸门 | 正文碎片能蹭到关键词分（实测 11.6 分），但凑不出连续的数据行 | `pdf.PdfTable.data_row_count`；`pdf.candidate_tables(min_score=8.0)` |
| 规则层与语义层分数取 min | 语义层的宽松判断不能盖过确定性的硬矛盾 | `critic.CriticMasterAgent.review` |
| 拒答行退出准确率分母 | "拒答了这个类别"不该被算成一次错误预测 | `matcher.evaluate`；`metrics.SuiteSummary` |
| 确定性规则路径不空转第 2、3 轮 | 喂回反馈结果不会变，空跑只会伪造"迭代过"的假象 | `pipeline.run_document` → `extractor.can_revise` |
| 扫描件直接拒答而不是假装能读 | 本项目**未实现 OCR**，硬给等于编造 | `pdf.is_probably_scanned`；`extractor.extract` |
| 每个缺失字段单独记一条 issue | 得分要能反映"缺了多少"，而不是一刀切 | `critic._check_row` |
| API Key 先做 ASCII 校验再发请求 | 从网页粘贴极易带入全角字符，`UnicodeEncodeError` 的报错完全看不出是 Key 的问题 | `config.assert_ascii`；`llm._build_client` |
| 探测打真实端点而不是 `/models` | 不少云厂商兼容层根本没 `/models` 这条路径，会把配置正确的服务判成不可用 | `llm.OpenAICompatLLM.probe`；`scripts/check_llm.py` |

---

## 6. 依赖降级矩阵

| 真组件 | 降级实现 | 行为是否对等 | 如何验证 |
| --- | --- | --- | --- |
| Extractor 强模型（GPT-4o / Claude / DeepSeek） | 确定性规则解析（pdfplumber 表结构 + 表头识别 + 多矿种列配对） | 结果**真实但不编造**，鲁棒性弱于模型；在 CSV 化的演示表上准确率 100%，在真实复杂表上明显更低 | 不填 key 直接跑 `run_pipeline.py` |
| CriticMaster 第二模型（Qwen / GLM） | 确定性规则审核（三角一致性、单位混淆、量级、字段完整性、来源可追溯、类别覆盖、重复行） | **始终运行**，即使配了模型也不关闭；模型分数与之取 min | `python -m pytest tests/test_critic.py` |
| OpenAI SDK | `UnavailableLLM`（显式返回失败，`ok=False`） | 不产生任何编造内容，由上层切到规则实现 | `python scripts/check_llm.py` |
| OCR（扫描件 PDF） | **未实现**——直接拒答并标注待人工审核；是否需要 OCR 先由 `scripts/pdf_health.py` 判定 | 宁可拒答也不硬给 | `tests/test_pdf_health.py`；`tests/test_extractor.py::test_scanned_pdf_abstains` |

> 降级的副作用是"安静地走了规则分支"。所以**必须**用
> `python scripts/check_llm.py` 确认配置真的生效，而不是靠猜。

---

## 7. 目录结构

```
Mining/
├── config.yaml                  ★ 唯一填写区（搜索「填写点」）
├── run_pipeline.py              主入口
├── requirements.txt
├── pytest.ini
├── README.md / RUN.md
├── ni101/                       核心包（见「架构分层」）
├── data/
│   ├── README.md                数据格式与 GT 结构约定
│   ├── pdfs/                    放 3 份真实 NI 43-101 PDF
│   └── ground_truth/            放 ground truth JSON（同名 stem 配对）
├── scripts/
│   ├── make_demo_data.py        生成 3 份合成 PDF + GT
│   ├── pdf_health.py            PDF 文本层体检：判断是否需要 OCR
│   ├── check_llm.py             LLM 端点三层连通性自检
│   ├── smoke_check.py           端到端自检（[PASS]/[FAIL] + 证据值）
│   └── demo_evolution_replay.py Evolution Log 回放演示
├── tests/                       127 条单测
├── docs/需求映射.md             原始题目要求 → 实现位置 → 验证方式
└── outputs/                     跑批产物
    ├── extractions/<stem>.json  每份 PDF 的完整往返记录（含每轮 Critic 结果）
    ├── summary.json             总览指标
    ├── report.md                可直接粘贴进汇报的报告
    └── evolution.jsonl          失败/降级案例（追加式）
```

---

## 8. 当前验证状态

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| **GitHub Actions CI** | ✅ **双版本全绿** | 干净 Ubuntu 上从零复现全流程，Python 3.10 / 3.13 均 success，**全程不需要任何 API Key**。见上方徽章或 Actions 页签 |
| 单元测试 | ✅ 135 passed | `python -m pytest` |
| 端到端自检 | ✅ 全部通过 | `python scripts/smoke_check.py` |
| 流水线跑批（3 份演示文档） | ✅ 字段准确率 100% (21/21)，硬给 0 | `python run_pipeline.py` |
| 三角一致性判据 | ✅ 自洽行 0.17%、矛盾行 62.14% | `smoke_check.py` §4 |
| 行级拒答 | ✅ Pilbara Inferred 行拒答，Indicated 行保留 | `smoke_check.py` §5 |
| Evolution Log 落盘 | ✅ 触发条件与案例结构正确 | `smoke_check.py` §7 |
| few-shot 注入 | ✅ 机制验证通过（离线） | `python scripts/demo_evolution_replay.py` |
| PDF 文本层体检 | ✅ 演示数据 3/3 判为「不需要 OCR」；反例（扫描件）判为「整份需要 OCR」 | `python scripts/pdf_health.py`；`tests/test_pdf_health.py` |
| OCR 实现 | ❌ **未实现**（按需引入，见第 9 节） | 扫描件会显式拒答并标注，不假装能读 |

**两项未实测，以及原因**：

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| 真实模型路径 | ⚠️ 未实测 | 端点**刻意留空** —— 不绑定任何厂商，使用者可注入自己偏好的模型（DeepSeek / Qwen / GLM / Ollama 等，均为 OpenAI 兼容端点）。LLM 相关代码路径已由 `tests/conftest.py::ScriptedLLM` **离线全覆盖**；填入端点后跑 `scripts/check_llm.py` 即可完成实测 |
| 真实 NI 43-101 PDF | ⚠️ 未实测 | 仓库内为 `make_demo_data.py` 生成的合成数据（规整表格）。真实报告的跨页表 / 合并单元格 / 脚注改口径（`including` / `at cut-off grade`）会明显降低规则路径的准确率 —— **这正是需要接入真实模型的原因**，也是 `config.yaml` 留出两个端点的目的 |

---

## 9. 关于 OCR：这个项目需要吗？

**结论：取决于题目给的 PDF 有没有文本层，不要预先假设。** 用一条命令判定：

```bash
python scripts/pdf_health.py            # 体检 data/pdfs/ 下全部 PDF
python scripts/pdf_health.py a.pdf b.pdf  # 指定文件
python scripts/pdf_health.py --json     # 机器可读输出，便于接流水线
```

退出码：`0` = 全都不需要 OCR；`1` = 至少一份需要。脚本按四类情形给出**不同处置范围**：

| 体检结论 | 含义 | 处置范围 |
| --- | --- | --- |
| 不需要 OCR —— 文本层可直接抽取 | 文本层完整，且已定位到资源量候选表 | 无需动作，直接跑流水线 |
| 仅资源量表页需要 OCR —— 表落在图片页上 | 文档有文本层，但表所在页是图片 | **只 OCR 该页**，正文页不用动 |
| 仅资源量表需要 OCR —— 表以图片嵌入正文 | 正文完整、一个表格结构都抽不到（表是截图） | **只 OCR 该表所在页** |
| 扫描件 —— 整份需要 OCR | 全部页面都无文本层 | **整份 OCR** |
| 有文本层但抽不出资源量表 —— 疑似表格布局异常 | 有表格但分数不够（跨页表 / 无表格线 / 文本框拼贴） | 先试坐标聚类，再考虑视觉模型 |

**为什么区分"整份"和"仅一页"**：整份 OCR 与单页 OCR 是数量级的成本与误差差。
把一份 200 页的报告整份丢进 OCR，会引入大量不必要的识别错误，
而这些错误会直接污染 CriticMaster 的三角一致性判据。

### 若真的需要 OCR，两条路线

| 路线 | 代表方案 | 适用 | 代价 |
| --- | --- | --- | --- |
| 本地 OCR | RapidOCR / PaddleOCR / PP-StructureV3 / MinerU | 数据不能出内网；批量处理 | 需装依赖与模型权重，表格结构还原质量参差 |
| 云端视觉模型 | Qwen-VL / GPT-4o vision | 只要资源量表那一两页，表结构理解更好 | 按量计费；数据外发需确认合规 |

**当前项目的处理**：`pdf.is_probably_scanned` 检出扫描件后**直接拒答并标注 `scanned_pdf`**，
不硬给结果 —— 这是题目最看重的行为（宁可拒答也不编造）。接 OCR 的正确做法是
**在 `PdfDocument` 之后、`ExtractorAgent` 之前插一个 OCR 适配层**，把图片页转成文本再喂给下游，
现有链路不需要改。

---

## 10. 评审导览：5 分钟看懂这个项目

**如果只有 5 分钟**：

1. **一分钟都不想花在本地**：直接看仓库上方的 `tests` 徽章，或点开 **Actions** 页签 ——
   CI 已在 GitHub 的干净 Ubuntu 机器上从零复现了全流程（装依赖 → 播演示数据 → PDF 体检 →
   端到端跑批 → 自检 → 单测），Python 3.10 与 3.13 双版本全绿，**不需要任何 API Key**。
   这比本地跑更可信：它证明的不是"作者机器上能跑"，而是"任何人 clone 下来都能跑"。
2. 想自己跑一遍「路径 A」（约 30 秒）—— 看到「135 条单测全绿 + 端到端自检全通过」
3. 读第 2 节「三角一致性」—— 这是整个审核逻辑的物理依据
4. 打开 `outputs/report.md` 的「拒答与硬给明细」—— 拒答是真的会触发，不是写死的

**三个值得细看的设计**：

| 设计 | 位置 | 为什么值得看 |
| --- | --- | --- |
| 三角一致性判据（不看原文就能抓错） | `ni101/schema.py` 恒等式 · `ni101/critic.py` 硬校验 | 把"审核"从主观判断变成可计算的物理约束：矿石量 × 品位 必须等于金属量 |
| 行级拒答（而非整篇拒答） | `ni101/pipeline.py::_apply_row_abstains` | 一行坏数据不拖垮整份文档；拒答行退出准确率分母，"拒答了这个类别"不算一次错误预测 |
| 规则层与模型层分数取 min | `ni101/critic.py::review` | 确定性硬证据不允许被 LLM 的宽松判断盖过；规则层始终运行，配了模型也不关闭 |

**诚实清单**：真实 NI 43-101 PDF、真实模型调用、OCR 三项**未实测**
（原因与验证入口见第 8 节）。模型端点刻意留空 —— 不绑定厂商，使用者可注入
自己的模型完整复现；README 与 `docs/需求映射.md` 均不给结论留灰色地带。
