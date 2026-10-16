"""质量评估节点（Critic 层）

从实体准确率、关系准确率、图连通性、图平衡性四维评估知识图谱质量。

为避免超大 knowledge_graph 超出 LLM 上下文，先对输入做轻量压缩：
去除节点/边中的空占位属性与超长文本，仅保留评估所需的量化信息。
"""
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import CriticInput, CriticOutput
from utils.llm_helper import invoke_llm, extract_json


def _compact_kg(knowledge_graph: dict) -> dict:
    """压缩知识图谱供质量评估：精简节点/边/属性，控制传给 LLM 的 token 量。"""
    if not isinstance(knowledge_graph, dict):
        return knowledge_graph or {}
    kg = dict(knowledge_graph)

    def _trim(s: str, limit: int = 200) -> str:
        s = str(s or "")
        return s if len(s) <= limit else s[:limit] + "…"

    def _compact_attrs(attrs: object) -> dict:
        if not isinstance(attrs, dict):
            return {}
        out = {}
        for k, v in attrs.items():
            s = str(v or "").strip()
            if not s:
                continue  # 丢弃空占位属性
            out[str(k)] = _trim(s, 80)
        return out

    nodes = kg.get("nodes") or []
    compact_nodes = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        compact_nodes.append({
            "id": str(n.get("id", "")),
            "name": _trim(n.get("name", ""), 30),
            "type": str(n.get("type", "unknown")),
            "definition": _trim(n.get("definition", ""), 90),
            "importance": n.get("importance", 3),
            "source_sections": (n.get("source_sections") or [])[:5],
            "attributes": _compact_attrs(n.get("attributes")),
        })
    kg["nodes"] = compact_nodes

    edges = kg.get("edges") or []
    compact_edges = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        prov = e.get("provenance") or {}
        compact_edges.append({
            "source": str(e.get("source", "")),
            "target": str(e.get("target", "")),
            "type": str(e.get("type", "")),
            "strength": float(e.get("strength", 0) or 0),
            "source_section": _trim(e.get("source_section", ""), 30),
            "provenance": {
                "supporting_text": _trim(prov.get("supporting_text", ""), 60),
                "extraction_method": prov.get("extraction_method", "explicit"),
            },
        })
    kg["edges"] = compact_edges
    return kg


def critic_node(
    state: CriticInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> CriticOutput:
    """
    title: 质量评估
    desc: 对构建的知识图谱进行四维量化评分，输出质量报告与回滚/改进建议。
    integrations: 大语言模型
    """
    ctx = runtime.context

    params = {
        "knowledge_graph": _compact_kg(state.knowledge_graph or {}),
        "task_contract": state.task_contract or {},
        "execution_plan": state.execution_plan or {},
    }
    resp_text = invoke_llm(runtime, config, render_params=params)
    parsed = extract_json(resp_text)

    quality_report = parsed.get("quality_report")
    if not isinstance(quality_report, dict) or not quality_report:
        quality_report = parsed
    quality_report.setdefault("decision", "conditional_pass")
    quality_report.setdefault("overall_score", 0.0)
    return CriticOutput(quality_report=quality_report)
