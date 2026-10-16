"""流程编排 Agent 节点（Planner 层 N4）

根据任务合约为本次构建产出执行计划 execution_plan。
执行计划不只包含模块与质量门，还包含领域自适应配置 domain_profile 与
下发给各 Worker 的领域适应性 prompt（worker_prompts）。
"""
import uuid
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import FlowOrchestratorInput, FlowOrchestratorOutput
from utils.llm_helper import invoke_llm, extract_json


def flow_orchestrator_node(
    state: FlowOrchestratorInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> FlowOrchestratorOutput:
    """
    title: 流程编排 Agent
    desc: 依据任务合约产出执行计划：激活模块与顺序/并行、模型路由、质量门与质量策略，并生成领域自适应本体配置 domain_profile 与下发给各 Worker 的领域适应性 prompt。
    integrations: 大语言模型
    """
    ctx = runtime.context

    params = {"task_contract": state.task_contract or {}, "has_files": bool(state.files)}
    resp_text = invoke_llm(runtime, config, render_params=params)
    parsed = extract_json(resp_text)

    execution_plan = parsed.get("execution_plan")
    if not isinstance(execution_plan, dict) or not execution_plan:
        execution_plan = parsed

    execution_plan.setdefault("plan_id", str(uuid.uuid4()))
    execution_plan.setdefault("task_id", str(state.task_contract.get("task_id", "")))
    execution_plan.setdefault("complexity_level", "moderate")
    execution_plan.setdefault("activated_modules", [])
    execution_plan.setdefault("quality_gates", [])
    execution_plan.setdefault("quality_policy", {
        "correctness_over_accuracy": False, "connectivity_priority": False,
        "enable_type_exploration": False, "dynamic_update": False,
        "threshold": 0.8, "sampling_rate": 0.1,
    })
    execution_plan.setdefault("human_checkpoints", [])
    # 多轮确认回写：若任务合约中已携带用户确认的候选本体（candidate_ontology 节点/关系/属性完整），
    # 则优先采用已确认本体作为 domain_profile，避免 Planner 每轮重新生成本体导致候选漂移。
    _tc = state.task_contract or {}
    _cand = _tc.get("candidate_ontology") or {}
    _cand_nodes_confirmed = [
        x for x in (_cand.get("node_types") or []) if isinstance(x, dict) and x.get("type_name")
    ]
    _cand_rels_confirmed = [
        x for x in (_cand.get("relation_types") or []) if isinstance(x, dict) and x.get("type_name")
    ]
    # 仅当"用户已明确确认"（clarity 非低 + 候选本体非空）时才采用已确认本体
    _json_clarity = _tc.get("clarity") or {}
    _sufficient = str(_json_clarity.get("info_sufficiency_level", "")).lower() != "low"

    # 判断是否使用 LLM 已生成本体（可保留其语义补充），否则走候选/兜底
    _llm_dp = execution_plan.get("domain_profile")
    _has_llm_dp = isinstance(_llm_dp, dict) and bool(_llm_dp.get("node_types") or _llm_dp.get("relation_types"))
    _use_confirmed = bool(_cand_nodes_confirmed and _cand_rels_confirmed and _sufficient)

    _ds = _tc.get("downstream_task") or {}
    if _use_confirmed:
        # 已确认本体回写（权威，覆盖 LLM 可能漂移的类型定义）
        execution_plan["domain_profile"] = {
            "domain": _tc.get("domain") or {"primary": "", "sub_domains": []},
            "downstream_task_summary": {
                "category": _ds.get("category", "unknown"),
                "requirements": _ds.get("requirements", []),
                "quality_priority": _ds.get("quality_priority", "balanced"),
            },
            "node_types": _cand_nodes_confirmed,
            "relation_types": _cand_rels_confirmed,
            "node_attribute_schema": _cand.get("node_attribute_schema", {})
            or (_llm_dp.get("node_attribute_schema") or {}),
            "domain_glossary": _cand.get("domain_glossary", [])
            or (_llm_dp.get("domain_glossary") or []),
            "entity_boundary": {
                "node_eligibility": [],
                "excluded_classes": [],
                "excluded_phrases": [],
                "repurpose_rule": "",
            },
        }
    elif _has_llm_dp:
        pass  # 保留 LLM 已生成本体
    elif "domain_profile" not in execution_plan or not isinstance(execution_plan.get("domain_profile"), dict):
        execution_plan["domain_profile"] = {
            "domain": _tc.get("domain") or {"primary": "", "sub_domains": []},
            "downstream_task_summary": {
                "category": _ds.get("category", "unknown"),
                "requirements": _ds.get("requirements", []),
                "quality_priority": _ds.get("quality_priority", "balanced"),
            },
            "node_types": _cand.get("node_types", []),
            "relation_types": _cand.get("relation_types", []),
            "node_attribute_schema": _cand.get("node_attribute_schema", {}),
            "domain_glossary": _cand.get("domain_glossary", []),
            "entity_boundary": {
                "node_eligibility": [],
                "excluded_classes": [],
                "excluded_phrases": [],
                "repurpose_rule": "",
            },
        }
    # 实体资格判定：确保 domain_profile.entity_boundary 结构完整（node_eligibility/excluded_classes/excluded_phrases/repurpose_rule）
    _dp = execution_plan.get("domain_profile") or {}
    if not isinstance(_dp.get("entity_boundary"), dict):
        _dp["entity_boundary"] = {
            "node_eligibility": [],
            "excluded_classes": [],
            "excluded_phrases": [],
            "repurpose_rule": "",
        }
    else:
        _eb = _dp["entity_boundary"]
        _eb.setdefault("node_eligibility", [])
        _eb.setdefault("excluded_classes", [])
        _eb.setdefault("excluded_phrases", [])
        _eb.setdefault("repurpose_rule", "")
    if "worker_prompts" not in execution_plan or not isinstance(execution_plan.get("worker_prompts"), dict):
        execution_plan["worker_prompts"] = {}
    # 若 LLM 未下发实体抽取 prompt，则从领域本体+实体资格判定组装一个含边界/排除规则的兜底抽取提示词
    _wp = execution_plan.get("worker_prompts") or {}
    if not (_wp.get("entity_extract_prompt") or "").strip():
        _dp2 = execution_plan.get("domain_profile") or {}
        _eb2 = _dp2.get("entity_boundary") or {}
        _excl = _eb2.get("excluded_phrases") or []
        _nt = [x.get("type_name", "") for x in (_dp2.get("node_types") or []) if isinstance(x, dict)]
        _gloss = [x.get("term", "") for x in (_dp2.get("domain_glossary") or []) if isinstance(x, dict)]
        _pp = "你是领域实体抽取 Worker。请依据以下节点类型与实体资格判定抽取实体。\n"
        if _nt:
            _pp += "允许入库的节点类型：" + "、".join(_nt) + "。\n"
        if _excl:
            _pp += "以下表述类别的对象不具备节点实例资格，禁止抽取为节点（仅作上下文背景）：" + "、".join(_excl) + "。\n"
        if _gloss:
            _pp += "领域参考词表：" + "、".join(_gloss) + "。\n"
        _pp += "优先抽取可入库的领域实体；若某提及仅属于排除类别或无足够类型归属，则不要输出为节点。"
        execution_plan["worker_prompts"]["entity_extract_prompt"] = _pp
    execution_plan.setdefault("model_routing", {"large_tasks": ["W3", "W4", "W7"], "medium_tasks": ["W2", "W5", "W6", "W8", "W9"], "small_tasks": ["W1", "W10"]})

    # 数据来源模式：无源文档且 LLM 未显式声明时，默认走联网检索，并基于领域/下游任务生成基础检索词
    if "sources" not in execution_plan or not isinstance(execution_plan.get("sources"), dict):
        _tc = state.task_contract or {}
        _ds = _tc.get("downstream_task") or {}
        _dm = _tc.get("domain") or {}
        _primary = _dm.get("primary", "") if isinstance(_dm, dict) else str(_dm)
        _keywords = _ds.get("keywords") or _tc.get("keywords") or []
        _queries = []
        if _primary:
            _base = f"{_primary} 领域知识"
            _queries.append({"query": _base, "purpose": "获取该领域的整体概念与知识结构"})
        for kw in (_keywords or [])[:5]:
            if isinstance(kw, str) and kw.strip():
                _queries.append({"query": kw.strip(), "purpose": "补充该主题的具体知识"})
        execution_plan["sources"] = {
            "mode": "documents" if bool(state.files) else "web_search",
            "reason": "无源文档输入，需通过联网检索补充领域资料" if not state.files else "已提供源文档，直接基于文档构建",
            "search_queries": _queries,
        }

    return FlowOrchestratorOutput(execution_plan=execution_plan)
