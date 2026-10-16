"""关系抽取节点（Worker 层 W4）

组内抽取实体间的语义关系，并识别孤立实体。
依据执行计划中的领域关系类型（relation_types）与
领域关系抽取 prompt（worker_prompts.relation_extract_prompt）做领域适应性抽取。
"""
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import RelationExtractInput, RelationExtractOutput
from utils.llm_helper import invoke_llm, extract_json


def relation_extract_node(
    state: RelationExtractInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> RelationExtractOutput:
    """
    title: 关系抽取
    desc: 依据领域关系类型与领域关系抽取prompt，在实体间抽取语义（教学）关系，识别孤立实体并补全建议关系。
    integrations: 大语言模型
    """
    ctx = runtime.context

    execution_plan = state.execution_plan or {}
    domain_profile = execution_plan.get("domain_profile") or {}
    worker_prompts = execution_plan.get("worker_prompts") or {}
    relation_prompt = worker_prompts.get("relation_extract_prompt", "")

    params = {
        "text_content": state.text_content or "",
        "entities": state.entities or [],
        "task_contract": state.task_contract or {},
        "execution_plan": execution_plan,
        "domain_profile": domain_profile,
        "relation_domain_prompt": relation_prompt,
    }
    resp_text = invoke_llm(runtime, config, render_params=params)
    parsed = extract_json(resp_text)

    relations = parsed.get("relations")
    isolated = parsed.get("isolated_entities", [])
    if not isinstance(relations, list):
        relations = []
    if not isinstance(isolated, list):
        isolated = []
    return RelationExtractOutput(relations=relations, isolated_entities=isolated)
