"""设计确认层节点（Planner 层）

在流程编排产出执行计划后，将 domain_profile（节点类型/关系类型/节点属性
/领域参考词表）整理为结构化、可读的领域设计方案，供用户确认是否符合需求。

该层是"计划完成后与用户确认设计"的关键一环：用户可查看并确认/调整
节点类型、关系类型、节点属性后，再进入 Worker 抽取阶段。
"""
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import DesignReviewInput, DesignReviewOutput
from utils.llm_helper import extract_json


def _safe_list(data, key: str, default=None) -> list:
    if default is None:
        default = []
    if not isinstance(data, dict):
        return default
    val = data.get(key, default)
    if isinstance(val, list):
        return val
    return default


def design_review_node(
    state: DesignReviewInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> DesignReviewOutput:
    """
    title: 领域设计确认
    desc: 将任务解析与流程编排生成的知识图谱设计方案（节点类型/关系类型/节点属性/领域词表）整理为可读提案，供用户确认是否符合需求。
    """
    ctx = runtime.context

    task_contract = state.task_contract or {}
    execution_plan = state.execution_plan or {}

    # 任务合约中的候选本体
    cand = task_contract.get("candidate_ontology") or {}
    cand_types = _safe_list(cand, "entity_types")
    cand_rels = _safe_list(cand, "relation_types")

    # 执行计划中收敛后的领域本体（权威）
    dp = execution_plan.get("domain_profile") or {}
    node_types = _safe_list(dp, "node_types") or cand_types
    relation_types = _safe_list(dp, "relation_types") or cand_rels
    node_attribute_schema = dp.get("node_attribute_schema") or cand.get("node_attribute_schema") or {}
    glossary = _safe_list(dp, "domain_glossary") or _safe_list(cand, "glossary")

    # 下游任务与质量侧重
    downstream = (task_contract.get("downstream_task") or {})
    domain = (task_contract.get("domain") or {})
    primary_domain = domain.get("primary") or "通用"

    # 信息充分度与是否需要追问
    clarity = task_contract.get("clarity") or {}
    needs_clarification = bool(
        clarity.get("needs_human_confirmation")
        or clarity.get("info_sufficiency_level") == "low"
    )
    clar_items = clarity.get("clarification_items") or []

    proposal = {
        "domain": primary_domain,
        "downstream_task": {
            "category": downstream.get("type") or downstream.get("category") or "general",
            "requirements": downstream.get("requirements") or [],
            "quality_priority": downstream.get("quality_priority") or "",
        },
        "node_types": node_types,
        "relation_types": relation_types,
        "node_attribute_schema": node_attribute_schema if isinstance(node_attribute_schema, dict) else {},
        "domain_glossary": glossary[:30] if glossary else [],
        "status": "need_confirm" if needs_clarification else "designed",
        "needs_user_confirmation": needs_clarification,
        "confirmation_items": clar_items,
        "activated_modules": (_safe_list(execution_plan, "activated_modules")
                              or (execution_plan.get("execution_plan") or {}).get("activated_modules", [])),
    }

    # 生成便于用户确认的自然语言说明
    type_names = [t.get("type_name") for t in node_types if isinstance(t, dict)]
    rel_names = [r.get("type_name") or r.get("relation_type") for r in relation_types if isinstance(r, dict)]
    proposal["render"] = (
        f"建议为「{primary_domain}」构建知识图谱，设计如下：\n"
        f"- 节点类型：{('、'.join(type_names)) if type_names else '（待定）'}\n"
        f"- 关系类型：{('、'.join(rel_names)) if rel_names else '（待定）'}\n"
        f"- 节点属性：{('、'.join(list(node_attribute_schema.keys()))) if isinstance(node_attribute_schema, dict) and node_attribute_schema else '（待定）'}"
    )
    if needs_clarification:
        proposal["render"] += "\n（信息不足，建议向用户追问以下项后确认设计）"

    return DesignReviewOutput(design_proposal=proposal)
