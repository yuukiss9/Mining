# Ground Truth JSON 放这里

- 文件名与 PDF 的 stem 对应，例如：
  `data/pdfs/newmont_ni43-101.pdf` → `data/ground_truth/newmont_ni43-101.json`
- **本目录不进版本库**（`.gitignore` 已排除 `*.json`）
- 目录为空时：跑 `python scripts/make_demo_data.py` 生成演示用的 GT

JSON 结构约定（含 `expect_abstain` 标记的含义）见 `../README.md`。
