# 学科知识图谱自动构建

这是本科毕设阶段的课程知识图谱实现，面向有权使用的 PDF 课件或教材章节。当前目录保留处理代码和一个人工编写的 JSON 结构示例；原始论文、答辩材料、课程文件、人工标注集和运行输出不随仓库发布。

## 流程

```text
PDF → OCR/文本提取 → 文本预处理 → 实体与关系抽取 → 匹配去重
    → 定义补全 → 类型规范化 → 新类型探索 → 图谱构建 → 质量评估
```

九个模块分别是 OCR、预处理、抽取、匹配、定义、规范化、类型探索、图谱构建和质量评估。实体/关系会保留定义、类型、章节或页码等信息；部分模块也会计算重要性、强度、覆盖率和结构指标。

## 环境与运行

需要 Python 3.10 或更高版本、可用的 OpenAI 兼容模型 API、Tesseract OCR 和 Poppler。扫描件通常需要 OCR；PDF 有可用文本层时也会尝试直接提取。依赖中的 PyTorch、Transformers 和句向量模型会增加安装体积。

```powershell
cd subject-kg
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑本机 `.env`，填入自己的模型配置。不要把真实密钥写入 Python 文件、命令行参数或 Git：

```dotenv
OPENAI_API_KEY=
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
TESSERACT_PATH=
POPPLER_PATH=
```

把有权处理的 PDF 放入 `data/slides/`，再运行：

```powershell
python run.py --all_files
```

处理单份文件时可用 `python run.py --input_file 文件名`，文件名不带 `.pdf`。如 OCR 程序不在系统 PATH 中，在 `.env` 设置 `TESSERACT_PATH`；需要指定 Poppler 时设置 `POPPLER_PATH`。生成的中间文件、图谱和数据库写入 `output/`，不会进入 Git。

## 示例与评估

[`examples/demo_graph.json`](examples/demo_graph.json) 展示实体、关系、定义、类型和来源字段，内容是手写的 TCP/IP 格式样例，不是模型运行结果。为了避免传播课程版权材料，仓库不提供原始 PDF、完整图谱或人工标注集。

项目论文记录了与人工标注和多种基线方法的对照，以及准确率、召回率、F1、类型匹配和图结构等维度。不同实验的样本、标注范围和计算口径不同；后续三门学科抽样迭代的“50%+ 到 90%+”数据也没有公开逐条标注。不要把这些历史数据与通用版案例的 Critic 质量分直接比较。要复算指标，需要准备有使用权的语料、人工金标准、模型配置和明确的评估口径。

## 限制

- 抽取结果会受输入质量、OCR、提示词、模型和采样参数影响；重要事实需对照原文核验。
- 当前实现面向课程材料流程，不等于可直接用于所有行业；切换领域需要调整知识类型、关系策略和评估集。
- 外部模型调用会发送处理文本到所配置的服务，并可能产生费用。不要处理未获授权的个人或内部数据。
- 示例数据不构成专业建议，也不代表原论文实验结果。
