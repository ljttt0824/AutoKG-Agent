"""类型标准化/探索节点（Worker 层 W9）

将实体/关系类型映射到标准类型，并探索提出新实体类型。
"""
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import TypeStandardInput, TypeStandardOutput
from utils.llm_helper import invoke_llm, extract_json


def type_standard_node(
    state: TypeStandardInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> TypeStandardOutput:
    """
    title: 类型标准化
    desc: 将实体/关系类型映射到标准类型，并在必要时提出质量达标的新实体类型建议。
    integrations: 大语言模型
    """
    ctx = runtime.context

    params = {
        "entity_definitions": state.entity_definitions or [],
        "relation_definitions": state.relation_definitions or [],
        "task_contract": state.task_contract or {},
    }
    resp_text = invoke_llm(runtime, config, render_params=params)
    parsed = extract_json(resp_text)

    return TypeStandardOutput(type_standardization=parsed)
