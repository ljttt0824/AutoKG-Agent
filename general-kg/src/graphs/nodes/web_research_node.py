"""联网检索节点（数据来源层）

当任务无源文档输入或依赖实时/外部资料时，依据执行计划中的 sources.search_queries
进行联网检索，将检索结果的 AI 摘要 / 网页片段聚合成研究文本 research_text，
作为下游实体/关系抽取的源文本（与文档文本合并）。
"""
import logging
from typing import List
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from coze_coding_dev_sdk import SearchClient
from graphs.state import WebResearchInput, WebResearchOutput


def _safe_str(val: object) -> str:
    """将任意检索字段安全转换为字符串"""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    return str(val)


def _aggregate_single_query(client, query: str, count: int) -> str:
    """针对单个检索词执行搜索并聚合可读文本（优先快速模式）"""
    ctx_logger = logging.getLogger("graph.web_research")
    response = None
    # 优先使用不含 AI 摘要的快速检索，降低整体耗时
    try:
        response = client.web_search(query=query, count=count, need_summary=False)
    except Exception as exc:  # noqa: BLE001
        ctx_logger.warning("检索词[%s]失败: %s", query, exc)
    if response is None or not (getattr(response, "web_items", None) or []):
        # 兜底：带 AI 摘要
        try:
            response = client.web_search_with_summary(query=query, count=count)
        except Exception as exc2:  # noqa: BLE001
            ctx_logger.warning("检索词[%s]兜底检索也失败: %s", query, exc2)
            return ""

    blocks: List[str] = []
    items = getattr(response, "web_items", None) or []
    for item in items:
        title = _safe_str(getattr(item, "title", "")).strip()
        snippet = _safe_str(getattr(item, "snippet", "")).strip()
        content = _safe_str(getattr(item, "content", "")).strip()
        parts: List[str] = [f"- {title}" if title else "- (未命名条目)"]
        if snippet:
            parts.append(f"  {snippet[:200]}")
        if content:
            parts.append(f"  {content[:250]}")
        blocks.append("\n".join(parts))
    return "\n".join(blocks)


def web_research_node(
    state: WebResearchInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> WebResearchOutput:
    """
    title: 联网检索
    desc: 依据执行计划中的检索词表执行联网搜索，将 AI 摘要与网页片段聚合为研究文本，作为知识图谱构建的源文本（无文档输入时作为主源）。
    integrations: 联网搜索
    """
    ctx = runtime.context
    # 从执行计划的 sources.search_queries 提取检索词（LLM 生成的领域检索意图词表）
    queries_list: List[dict] = []
    sources = state.execution_plan.get("sources") if isinstance(state.execution_plan, dict) else {}
    if isinstance(sources, dict):
        _sq = sources.get("search_queries")
        if isinstance(_sq, list):
            queries_list = [q for q in _sq if isinstance(q, dict)]

    # 限制单次最多检索的查询数量，避免结果过载
    effective = queries_list[:2]

    aggregated: List[str] = []
    total_items = 0
    if not effective:
        logging.getLogger("graph").warning("联网检索节点未获取到检索词，跳过检索")
    try:
        client = SearchClient(ctx=ctx)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("graph").warning("初始化搜素客户端失败: %s", exc)
        return WebResearchOutput(research_text="", search_item_count=0)

    for q_entry in effective:
        query = _safe_str(q_entry.get("query")).strip()
        purpose = _safe_str(q_entry.get("purpose")).strip()
        if not query:
            continue
        header = f"【检索主题】{query}"
        if purpose:
            header += f"（目的：{purpose}）"
        block = _aggregate_single_query(client, query, count=3)
        if block.strip():
            aggregated.append(f"{header}\n{block}")
            total_items += 1

    research_text = "\n\n".join(aggregated).strip()
    # 控制研究文本规模，避免后续 LLM 抽取过慢/超长
    if len(research_text) > 5000:
        research_text = research_text[:5000]
    return WebResearchOutput(research_text=research_text, search_item_count=total_items)
