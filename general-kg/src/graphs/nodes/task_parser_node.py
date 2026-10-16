"""任务解析 Agent 节点（Planner 层 N2）

将用户自然语言需求解析为结构化任务合约 task_contract。
该合约不仅包含领域识别，还包含【下游任务分析】与【候选领域自适应本体】
（节点类型+属性、关系类型+业务语义、领域参考词表），并在信息不足时输出追问澄清项。
"""
import uuid
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import TaskParserInput, TaskParserOutput
from utils.llm_helper import invoke_llm, extract_json


def task_parser_node(
    state: TaskParserInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> TaskParserOutput:
    """
    title: 任务解析 Agent
    desc: 将用户自然语言构建需求解析为结构化任务合约：识别领域与下游任务，推导候选节点类型/属性/关系类型/领域参考词表，信息不足时自动生成追问澄清项。
    integrations: 大语言模型
    """
    ctx = runtime.context

    params = {
        "user_query": state.user_query or "",
        "domain_context": state.domain_context or "无",
        "existing_kg_id": state.existing_kg_id or "无",
        "files": state.files or [],
    }

    resp_text = invoke_llm(runtime, config, render_params=params)
    parsed = extract_json(resp_text)

    task_contract = parsed.get("task_contract")
    if not isinstance(task_contract, dict) or not task_contract:
        task_contract = parsed

    # 兜底保证必填字段存在
    task_contract.setdefault("task_id", str(uuid.uuid4()))
    task_contract.setdefault("task_type", "build_new")
    task_contract.setdefault("overall_confidence", 0.5)
    task_contract.setdefault("risk_level", "medium")
    task_contract.setdefault("timeliness", "batch")
    task_contract.setdefault("fallback_template_used", False)
    if "domain" not in task_contract or not isinstance(task_contract["domain"], dict):
        task_contract["domain"] = {"primary": "", "sub_domains": [], "ontology_available": False, "ontology_match_score": 0.0}
    if "downstream_task" not in task_contract or not isinstance(task_contract.get("downstream_task"), dict):
        task_contract["downstream_task"] = {
            "category": "unknown", "description": "", "requirements": [],
            "quality_priority": "balanced", "needs_dynamic_update": False,
            "needs_type_exploration": False, "needs_source_trace": False,
        }
    if "candidate_ontology" not in task_contract or not isinstance(task_contract.get("candidate_ontology"), dict):
        task_contract["candidate_ontology"] = {
            "node_types": [], "relation_types": [], "node_attribute_schema": {}, "domain_glossary": [],
        }
    _cand = task_contract["candidate_ontology"]
    _cand.setdefault("node_types", [])
    _cand.setdefault("relation_types", [])
    _cand.setdefault("node_attribute_schema", {})
    _cand.setdefault("domain_glossary", [])
    # 兼容旧字段 entity_types/relation_types（下游 worker 可能仍引用）
    task_contract.setdefault("entity_types", [])
    task_contract.setdefault("relation_types", [])
    if "clarity" not in task_contract or not isinstance(task_contract.get("clarity"), dict):
        task_contract["clarity"] = {
            "info_sufficiency_level": "medium",
            "needs_human_confirmation": False,
            "clarification_items": [],
            "can_proceed_with_best_effort": True,
        }
    if "input_spec" not in task_contract or not isinstance(task_contract.get("input_spec"), dict):
        task_contract["input_spec"] = {"file_count": len(state.files or []), "file_types": [], "estimated_pages": 0, "language": "mixed"}

    return TaskParserOutput(task_contract=task_contract)
