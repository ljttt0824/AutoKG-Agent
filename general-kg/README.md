# AutoKG-Agent / 通用领域知识图谱多 Agent 系统

这是在 Coze Coding 环境中实现的 LangGraph 版本。它把用户目标、领域资料和下游用途交给 Planner 生成本次任务的领域模式，再由 Worker 构图、Critic 评估。教育课程是首个领域包，后续案例覆盖电商推荐、实验复盘和 PMO 岗位百科。

## 工作台展示与 KG-RAG 衔接

[计算机网络课程公开工作台](https://ljttt0824.github.io/AutoKG-Agent/) 用脱敏静态数据展示 3 个已保存的问答溯源案例和 2 条学习路径推荐，可查看来源页码与案例质量状态。它适合项目展示；GitHub Pages 不会在线查询课程材料、调用模型 API 或接收用户文件。

独立 KG-RAG 系统可通过 KGX v1 导入本构图系统生成的图谱快照与来源定位信息，再由工作台调用其问答接口。整体设计方案中，查询服务先判断问题类型，再选择 BM25 关键词检索、向量召回、图遍历或社区摘要；通过 RRF 融合候选，并把实体重要性、关系强度、抽取置信度、路径长度和证据质量纳入排序，在 Token 预算内选择有来源支持的子图。模型生成后按声明核验引用，并提供回到原文页码或片段的溯源；学习路径则基于先修关系图计算。可审计过程以检索计划、图路径和证据账本呈现，不请求或展示模型隐藏思维文本。

方案中的服务端目标选型为 FastAPI + LangGraph、Neo4j 图存储、PostgreSQL/pgvector 元数据与向量检索、S3 兼容对象存储。公开工作台目前是静态演示，完整 KG-RAG 服务、来源材料和导入知识库不包含在此次 README 更新中；实时问答需单独部署后端并配置接口。

## 计算机网络课程图谱基线

以下为导入 KG-RAG 知识库的图谱快照统计。来源材料名称和内部证据 ID 不公开。

| 节点 | 关系 | 来源材料数 | 已记录的质量状态 |
| ---: | ---: | ---: | --- |
| 2,840 | 7,172 | 15 | 导入时关系端点完整性检查通过；Critic 质量分未记录，状态为待人工复核 |

因此该快照不报告整体质量分、Precision/Recall 或专家标注准确率。这里的导入检查结果也不等同于通用版 Planner-Worker-Critic 案例分数。

## 架构

```text
Planner: task_parser → flow_orchestrator
       → design_review → design_gate
       → source_gate → [web_research 可选]
Worker:  text_extract → entity_extract → relation_extract
       → entity_match / relation_match
       → entity_define / relation_define
       → type_standard → graph_build
Critic: critic → critic_decision → 放行或回滚重试（最多 2 次）
```

Planner 生成任务合约、执行计划、领域本体、关系/属性约束和 Worker 专属提示词。信息不足时，设计确认门返回 `confirmation_questions` 并暂停；没有源文件且计划需要外部资料时，来源路由进入联网研究。Worker 产出标准化图谱，Critic 按实体、关系、连通性和平衡性评估并给出决策。节点与状态字段见 [`docs/architecture.md`](docs/architecture.md) 和 `src/graphs/state.py`。

## 运行要求

端到端运行针对 Linux Coze Coding 工作区设计，需要 Python 3.12、Coze Coding SDK/Workload Identity、可用的大模型、必要时的搜索服务，以及对象存储配置。`pyproject.toml` 和 `uv.lock` 固定 Python 依赖版本，但依赖安装本身不能替代 Coze 身份、模型权限、搜索服务和对象存储配置。普通 Windows 虚拟环境无法提供这些平台服务。

在有上述平台能力的工作区中：

```bash
cd general-kg
uv sync --frozen
export COZE_WORKSPACE_PATH="$PWD"
bash scripts/local_run.sh -m flow -i '{"user_query":"为计算机网络课程构建用于复习问答的知识图谱，体现协议之间的依赖关系和来源章节","files":[]}'
```

带文档运行时，将 `files` 设为可访问文件对象，例如：

```json
{
  "user_query": "从课程讲义构建教学知识图谱，保留章节溯源和先修关系",
  "files": [
    {"url": "examples/sample_network_notes.txt", "file_type": "document"}
  ]
}
```

本地文件路径必须存在于运行进程可访问的工作目录中。脚本通过 Coze Workload Identity 加载项目运行环境；不要把凭据写入 `config/*.json`。示例文本和请求见 [`examples/`](examples/)。

启动 HTTP 服务可用 `bash scripts/http_run.sh -p 5000`，主图接口为 `POST /run`，健康检查为 `GET /health`。请求体与上面的 `GraphInput` 一致。返回值包含 `knowledge_graph`、`quality_report`、`design_proposal`、`gate_pending`、`confirmation_questions` 和 `decision`；图谱文件还可能通过对象存储返回链接。

## 已实现案例记录

| 场景 | 输入方式 | 节点 / 边 | 历史 Critic 分数 |
| --- | --- | ---: | ---: |
| 电商新生购物推荐 | 无文件，联网检索 | 27 / 40 | 0.86 |
| 实验复盘经验知识库 | 复盘文档 | 44 / 31 | 0.80 |
| PMO 岗位职责百科 | 岗位材料 | 13 / 22 | 0.746 |
| 计算机网络课程 | 课程文档 | 案例记录为通过，未发布原始计数 | - |

这些数字来自项目运行记录，未附原始企业材料、完整图谱或独立复核数据；Critic 分数是单次运行质量信号，不等同于人工金标准准确率，不能据此推断其他任务的效果。案例材料已从公开目录排除。

## 使用边界

- 该系统是知识整理和候选关系生成工具，不应未经人工审核用于高风险或自动化决策。
- LLM 会产生遗漏、误分类和无来源断言。核验图谱时应检查来源片段、页码和关系方向。
- 联网检索依赖外部服务，结果受时间、地区、服务权限和来源许可影响。
- 上传文档可能经 Coze、模型或搜索服务处理。只使用已获授权且符合组织数据规定的资料；提交前移除个人信息、客户数据和机密内容。
- 模型、搜索、Coze、数据库和对象存储可能收费。仓库不包含密钥或可用服务账户。
- 当前代码与 Coze 运行时耦合。仅凭公开代码和 Python 依赖无法在任意本机复现全部服务能力；分数也受模型、提示词、检索和数据版本影响。
- 仓库未声明代码开源许可证。第三方文档、业务材料和依赖按其原许可管理。
