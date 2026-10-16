"""主图编排（Planner 层 + 设计确认层 + 数据来源路由 + 知识图谱构建子图调用）

DAG 结构：
  task_parser -> flow_orchestrator -> design_review -> design_gate
      ├─ need_confirmation -> END（返回追问澄清项，等待用户补充后重建）
      └─ proceed -> source_gate
            ├─ 需联网检索 -> web_research -> knowledge_build(调用子图) -> END
            └─ 直接文档构建 -> knowledge_build(调用子图) -> END
"""
from langgraph.graph import StateGraph, END
from graphs.state import (
    GlobalState,
    GraphInput,
    GraphOutput,
    DesignGateInput,
    SourceGateOutput,
)
from graphs.nodes.task_parser_node import task_parser_node
from graphs.nodes.flow_orchestrator_node import flow_orchestrator_node
from graphs.nodes.design_review_node import design_review_node
from graphs.nodes.design_gate_node import design_gate_node
from graphs.nodes.source_gate_node import source_gate_node
from graphs.nodes.web_research_node import web_research_node
from graphs.nodes.knowledge_build_node import knowledge_build_node


def design_gate(state: DesignGateInput) -> str:
    """
    title: 设计确认分支
    desc: 依据设计确认门决策决定放行构图或暂停追问。返回中文分支名。
    """
    if state.gate_decision == "need_confirmation":
        return "等待用户确认"
    return "开始构建"


def source_gate(state: SourceGateOutput) -> str:
    """
    title: 数据来源分支
    desc: 依据数据来源路由决策，决定进入联网检索还是直接基于文档构建。返回中文分支名。
    """
    if state.source_decision == "web":
        return "需联网检索"
    return "直接构建"


# 主图：全局状态 + 主图出入参
builder = StateGraph(GlobalState, input_schema=GraphInput, output_schema=GraphOutput)

builder.add_node("task_parser", task_parser_node, metadata={"type": "agent", "llm_cfg": "config/task_parser_llm_cfg.json"})
builder.add_node("flow_orchestrator", flow_orchestrator_node, metadata={"type": "agent", "llm_cfg": "config/flow_orchestrator_llm_cfg.json"})
builder.add_node("design_review", design_review_node, metadata={"type": "task"})
builder.add_node("design_gate", design_gate_node, metadata={"type": "condition"})
builder.add_node("source_gate", source_gate_node, metadata={"type": "condition"})
builder.add_node("web_research", web_research_node, metadata={"type": "task"})
builder.add_node("knowledge_build", knowledge_build_node, metadata={"type": "loopcond"})

builder.set_entry_point("task_parser")

builder.add_edge("task_parser", "flow_orchestrator")
builder.add_edge("flow_orchestrator", "design_review")
builder.add_edge("design_review", "design_gate")
builder.add_conditional_edges(
    source="design_gate",
    path=design_gate,
    path_map={
        "等待用户确认": END,
        "开始构建": "source_gate",
    },
)
builder.add_conditional_edges(
    source="source_gate",
    path=source_gate,
    path_map={
        "需联网检索": "web_research",
        "直接构建": "knowledge_build",
    },
)
builder.add_edge("web_research", "knowledge_build")
builder.add_edge("knowledge_build", END)

main_graph = builder.compile()
