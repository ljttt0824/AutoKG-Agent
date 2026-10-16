"""主图承载节点：调用知识图谱构建子图（Worker-Critic 执行层）

从子图获取最终图谱与质检报告，并将图谱上传到对象存储返回访问 URL。
"""
import json
import logging
import os
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from coze_coding_dev_sdk.s3 import S3SyncStorage
from graphs.state import KnowledgeBuildInput, KnowledgeBuildOutput
from graphs.loop_graph import subgraph


def knowledge_build_node(
    state: KnowledgeBuildInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> KnowledgeBuildOutput:
    """
    title: 知识图谱构建执行
    desc: 调用 Worker-Critic 执行层子图，完成实体/关系抽取、定义、标准化与质量评估，并将最终图谱上传对象存储。
    """
    ctx = runtime.context
    logger = logging.getLogger("graph.build")

    result = subgraph.invoke({
        "files": state.files or [],
        "task_contract": state.task_contract or {},
        "execution_plan": state.execution_plan or {},
        "research_text": state.research_text or "",
    })

    kg = result.get("knowledge_graph") or {}
    report = result.get("quality_report") or {}
    decision = result.get("critic_decision") or ""

    # 上传对象存储
    graph_url = ""
    try:
        storage = S3SyncStorage(
            endpoint_url=os.getenv("COZE_BUCKET_ENDPOINT_URL"),
            access_key="",
            secret_key="",
            bucket_name=os.getenv("COZE_BUCKET_NAME"),
            region="cn-beijing",
        )
        key = storage.upload_file(
            file_content=json.dumps(kg, ensure_ascii=False).encode("utf-8"),
            file_name="knowledge_graph.json",
            content_type="application/json",
        )
        graph_url = storage.generate_presigned_url(key=key, expire_time=3600)
    except Exception as exc:  # noqa: BLE001
        logger.warning("图谱上传对象存储失败: %s", exc)

    metrics = kg.get("graph_metrics") or {}
    node_count = metrics.get("node_count", len(kg.get("nodes", [])))
    edge_count = metrics.get("edge_count", len(kg.get("edges", [])))
    score = report.get("overall_score")
    summary = (
        f"知识图谱构建完成：{node_count} 个节点 / {edge_count} 条边，"
        f"质量评分 {score if score is not None else 'N/A'}，决策 {decision}。"
    )

    return KnowledgeBuildOutput(
        knowledge_graph=kg,
        quality_report=report,
        decision=decision,
        graph_file_url=graph_url,
        summary=summary,
    )
