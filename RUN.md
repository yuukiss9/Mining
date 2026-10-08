# RUN.md —— 5 分钟跑起来

## 0. 前置

- Python 3.10+
- 依赖：

```bash
git clone <本仓库地址>
cd <仓库目录>
pip install -r requirements.txt
```

> **不想填 Key 也想看结果**？直接按 README 的「路径 A」执行
> （`make_demo_data.py` → `run_pipeline.py` → `smoke_check.py`），全流程约 30 秒。

---

## 1. 填 API Key（想跑真实模型时才有必要）

打开 `config.yaml`，搜索中文关键词 **「填写点」**，共 2 处：

| 填写点 | 位置 | 说明 |
| --- | --- | --- |
| **1 / 2** | `extractor.api_key` + `extractor.base_url` | Extractor Agent，用**强模型**（GPT-4o / Claude / DeepSeek） |
| **2 / 2** | `critic.api_key` + `critic.base_url` | CriticMaster Agent，**必须换一个厂商**（Qwen / GLM） |

`base_url` 常见取值：

```
DeepSeek        https://api.deepseek.com/v1
OpenAI          https://api.openai.com/v1
通义千问 Qwen    https://dashscope.aliyuncs.com/compatible-mode/v1
智谱 GLM        https://open.bigmodel.cn/api/paas/v4
月之暗面 Kimi    https://api.moonshot.cn/v1
本地 Ollama     http://localhost:11434/v1   (api_key 随便填 ollama)
```

不想把 key 落到文件里，也可以改用环境变量（优先级更高）：

```powershell
$env:NI101_EXTRACTOR_API_KEY="sk-xxx"
$env:NI101_EXTRACTOR_BASE_URL="https://api.deepseek.com/v1"
$env:NI101_CRITIC_API_KEY="sk-yyy"
$env:NI101_CRITIC_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
```

> **留空也能跑**：会自动降级为「规则抽取 + 规则审核」，结果真实但不编造，
> 只是抽取质量弱于真模型。适合先验证工程结构。

### 填完务必自检

```bash
python scripts/check_llm.py
```

分三层输出（配置 / 可达 / 调用）。降级设计最大的坑是**配置写错不会报错**，
只是安静地走规则分支——这个脚本就是为了让失败点无处可藏。

```bash
python scripts/check_llm.py --quick   # 只探配置与可达性，不消耗额度
```

---

## 2. 准备数据

### 2a. 没有真实 PDF 时：生成演示数据

```bash
python scripts/make_demo_data.py
```

生成 3 份结构逼真的合成 NI 43-101（Newmont / Barrick / Pilbara）+ 对应 ground truth。
其中刻意埋了两类考点：表尾 `Total` 合计行、以及 Pilbara 一行**原文自相矛盾**的金属量
（正确行为是行级拒答而不是硬给）。

### 2b. 有真实 PDF 时：直接放进去

```
data/pdfs/newmont_ni43-101.pdf          <- 题目给的 3 份真实 PDF
data/pdfs/barrick_ni43-101.pdf
data/pdfs/pilbara_minerals_ni43-101.pdf

data/ground_truth/newmont_ni43-101.json <- 同名 stem 配对
data/ground_truth/barrick_ni43-101.json
data/ground_truth/pilbara_minerals_ni43-101.json
```

JSON 结构约定见 `data/README.md`。若题目给的 GT 结构不同，只改
`ni101/matcher.py::load_ground_truth()` 一个函数即可。

### 2c. 放进去之后先体检：这份 PDF 需要 OCR 吗？

```bash
python scripts/pdf_health.py
```

一条命令回答"要不要引入 OCR"，并给出**处置范围**（整份 OCR / 只 OCR 一页 / 不用 OCR）。
退出码 `0` = 全都不需要；`1` = 至少一份需要。

**别跳过这步**：真实 NI 43-101 绝大多数是咨询公司排版的文本型 PDF（不需要 OCR），
但也存在资源量表被做成截图贴进正文的情况 —— 那时文本层再完整也抽不到表。
体检报告会直接指出是哪种。判定逻辑见 README 第 9 节。

---

## 3. 跑批

```bash
python run_pipeline.py                      # 全部 PDF
python run_pipeline.py --only pilbara       # 只跑文件名含该子串的
python run_pipeline.py --replay-evolution   # 用 evolution.jsonl 做 few-shot 复跑
python run_pipeline.py --quiet              # 只输出汇总
python run_pipeline.py --max-rounds 2 --pass-score 7.5   # 临时改流程参数
```

**退出码**：`0` = 没有硬给；`1` = 存在硬给（应拒答却给了数值）。
可以直接当 CI 闸门用。

产物：

```
outputs/extractions/<stem>.json   每份 PDF 的完整往返记录（含每轮 Critic 评分与问题清单）
outputs/summary.json              总览指标
outputs/report.md                 可直接粘贴进汇报的报告
outputs/evolution.jsonl           失败/降级案例（追加式）
```

---

## 4. 验证

```bash
python -m pytest              # 127 条单测
python scripts/smoke_check.py # 端到端自检，打印 [PASS]/[FAIL] + 证据值，退出码 = 失败项数
```

---

## 5. Evolution Log 回放（few-shot 改进）

```bash
python scripts/demo_evolution_replay.py
```

- **第一半**：统计 `outputs/evolution.jsonl` 里有多少条可用于 few-shot
  （需要 `predicted` 与 `corrected` 都非空，即"错的输出 + 对的输出"的对照）
- **第二半**：离线验证 few-shot **确实注入**到了 Extractor 的 prompt，
  并原样打印注入片段

> 边界说明：未配置真实模型时走的是确定性规则抽取，它不使用 prompt，
> 因此 few-shot 不会改变结果。要看到"分数真的提升"，先完成第 1 步填 key，
> 再跑 `python run_pipeline.py --replay-evolution`。

---

## 6. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 日志里出现「降级为规则模式」 | `config.yaml` 未填写或填写不完整 | 完成第 1 步，再跑 `scripts/check_llm.py` |
| `check_llm.py` 报 DNS 解析失败 | `base_url` 拼写错误 | 对照第 1 步的常见取值表 |
| `check_llm.py` 报 404 路径不存在 | `base_url` 少了 `/v1` 或版本前缀 | 补上版本前缀 |
| 报 `UnicodeEncodeError` / 「含有非 ASCII 字符」 | API Key 从网页粘贴时带入了全角字符或中文占位符 | 重新粘贴纯 ASCII 的 key |
| 所有文档都拒答，理由是「疑似扫描件」 | PDF 是图片型，无文本层 | 本项目**未实现 OCR**，需要先做 OCR 再放入 |
| 文档状态是 `partial` | 有行级拒答（原文自相矛盾的行） | 这是**正确行为**，看 `outputs/report.md` 的拒答明细 |
| 字段准确率低但状态是 `pass` | 不该发生——`run_pipeline.py` 会把这类文档标为「硬给文档」 | 看 `report.md` 表格里的 ⚠ 标记 |
| 想换 ground truth 结构 | 题目给的 GT 字段名不同 | 只改 `ni101/matcher.py::load_ground_truth()` |
