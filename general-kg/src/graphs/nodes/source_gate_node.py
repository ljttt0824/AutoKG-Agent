"""数据来源路由节点（condition）

判断当前任务是否需要联网检索补充源文本：
- 无源文档输入，或执行计划声明数据来源模式为 web_search/hybrid 且需外部资料 → web（联网检索）
- 已提供源文档且无外部依赖 → direct（直接基于文档构建）
"""
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import SourceGateInput, SourceGateOutput


def source_gate_node(
    state: SourceGateInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> SourceGateOutput:
    """
    title: 数据来源路由
    desc: 判断是否有源文档、执行计划是否需要联网检索，决定进入联网检索还是直接依赖文档构建。
    """
    ctx = runtime.context

    has_files = bool(state.files)
    sources = state.execution_plan.get("sources") if isinstance(state.execution_plan, dict) else {}
    mode = sources.get("mode") if isinstance(sources, dict) else ""

    if (not has_files) or mode in ("web_search", "hybrid"):
        return SourceGateOutput(
            source_decision="web",
            source_reason="无源文档输入或执行计划声明需联网检索外部资料，进入联网检索补充源文本",
        )
    return SourceGateOutput(
        source_decision="direct",
        source_reason="已提供源文档且无外部依赖，直接基于文档构建",
    )
