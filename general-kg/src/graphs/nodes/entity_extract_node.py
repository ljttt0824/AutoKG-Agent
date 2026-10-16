"""实体抽取节点（Worker 层 W3）

从预处理文本中抽取领域实体，含类型、权重、置信度。
依据执行计划中的领域自适应本体（node_types/domain_glossary）与
领域抽取 prompt（worker_prompts.entity_extract_prompt）做领域适应性抽取。
"""
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import EntityExtractInput, EntityExtractOutput
from utils.llm_helper import invoke_llm, extract_json


def entity_extract_node(
    state: EntityExtractInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> EntityExtractOutput:
    """
    title: 实体抽取
    desc: 依据领域自适应本体（节点类型/参考词表/领域抽取prompt）从文本中抽取领域实体并归类。
    integrations: 大语言模型
    """
    ctx = runtime.context

    execution_plan = state.execution_plan or {}
    domain_profile = execution_plan.get("domain_profile") or {}
    worker_prompts = execution_plan.get("worker_prompts") or {}
    entity_prompt = worker_prompts.get("entity_extract_prompt", "")

    params = {
        "text_content": state.text_content or "",
        "task_contract": state.task_contract or {},
        "execution_plan": execution_plan,
        "domain_profile": domain_profile,
        "entity_domain_prompt": entity_prompt,
    }
    resp_text = invoke_llm(runtime, config, render_params=params)
    parsed = extract_json(resp_text)

    entities = parsed.get("entities")
    if not isinstance(entities, list):
        entities = []
    return EntityExtractOutput(entities=entities)
