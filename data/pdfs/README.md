# 待抽取的 NI 43-101 PDF 放这里

- 文件名随意，流水线自动扫描本目录下所有 `*.pdf`
- **本目录不进版本库**（`.gitignore` 已排除 `*.pdf`），真实报告不会被误提交
- 目录为空时：跑 `python scripts/make_demo_data.py` 生成合成演示数据
- 放入真实 PDF 后：**先跑 `python scripts/pdf_health.py` 体检文本层**，
  确认是否需要 OCR 以及处置范围（整份 / 仅个别页）

数据格式与 ground truth 约定见 `../README.md`。
