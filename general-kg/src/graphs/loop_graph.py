"""知识图谱构建执行层子图（Worker 层 + Critic 层）

子图承载从文档文本到最终知识图谱 + 质量评估的完整执行链，
并在 Critic 判定回滚时形成重试循环（子图内部允许闭环）。
"""
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from graphs.state import (
    GlobalState,
    SubGraphInput,
    SubGraphOutput,
    LoopRouteState,
)
from graphs.nodes.text_extract_node import text_extract_node
from graphs.nodes.entity_extract_node import entity_extract_node
from graphs.nodes.relation_extract_node import relation_extract_node
from graphs.nodes.entity_match_node import entity_match_node
from graphs.nodes.relation_match_node import relation_match_node
from graphs.nodes.entity_define_node import entity_define_node
from graphs.nodes.relation_define_node import relation_define_node
from graphs.nodes.type_standard_node import type_standard_node
from graphs.nodes.graph_build_node import graph_build_node
from graphs.nodes.critic_node import critic_node
from graphs.nodes.critic_decision_node import critic_decision_node, MAX_RETRY


def route_after_loop(state: LoopRouteState) -> str:
    """
    title: 回滚路由
    desc: 依据 Critic 决策与重试次数，判定是回滚重跑还是结束子图。
    """
    decision = state.critic_decision
    retry = int(state.retry_count or 0)
    if decision == "rollback" and retry <= MAX_RETRY:
        return "retry"
    return "complete"


# 子图：StateGraph 全局状态 + 子图出入参
builder = StateGraph(GlobalState, input_schema=SubGraphInput, output_schema=SubGraphOutput)

builder.add_node("text_extract", text_extract_node)
builder.add_node("entity_extract", entity_extract_node, metadata={"type": "agent", "llm_cfg": "config/entity_extract_llm_cfg.json"})
builder.add_node("relation_extract", relation_extract_node, metadata={"type": "agent", "llm_cfg": "config/relation_extract_llm_cfg.json"})
builder.add_node("entity_match", entity_match_node)
builder.add_node("relation_match", relation_match_node)
builder.add_node("entity_define", entity_define_node, metadata={"type": "agent", "llm_cfg": "config/entity_define_llm_cfg.json"})
builder.add_node("relation_define", relation_define_node, metadata={"type": "agent", "llm_cfg": "config/relation_define_llm_cfg.json"})
builder.add_node("type_standard", type_standard_node, metadata={"type": "agent", "llm_cfg": "config/type_standard_llm_cfg.json"})
builder.add_node("graph_build", graph_build_node)
builder.add_node("critic", critic_node, metadata={"type": "agent", "llm_cfg": "config/critic_llm_cfg.json"})
builder.add_node("critic_decision", critic_decision_node)

builder.set_entry_point("text_extract")

builder.add_edge("text_extract", "entity_extract")
builder.add_edge("entity_extract", "relation_extract")
builder.add_edge("relation_extract", "entity_match")
builder.add_edge("entity_match", "relation_match")
builder.add_edge("relation_match", "entity_define")
builder.add_edge("entity_define", "relation_define")
builder.add_edge("relation_define", "type_standard")
builder.add_edge("type_standard", "graph_build")
builder.add_edge("graph_build", "critic")
builder.add_edge("critic", "critic_decision")

# 回滚循环：critic_decision -> retry(回 entity_extract) / complete(结束)
builder.add_conditional_edges(
    source="critic_decision",
    path=route_after_loop,
    path_map={
        "retry": "entity_extract",
        "complete": END,
    },
)

subgraph = builder.compile()
