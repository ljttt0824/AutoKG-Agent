"""回滚决策节点（Critic 层）

根据质量报告决策与重试次数，确定是放行、重跑还是暂停人工审核。
"""
import logging
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import CriticDecisionInput, CriticDecisionOutput

MAX_RETRY = 2


def critic_decision_node(
    state: CriticDecisionInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> CriticDecisionOutput:
    """
    title: 回滚决策
    desc: 依据质量评估报告的决策与已重试次数，判定流程是放行、回滚重跑还是暂停人工审核。
    """
    ctx = runtime.context
    logger = logging.getLogger("graph.critic")

    report = state.quality_report or {}
    decision = str(report.get("decision", "conditional_pass"))
    retry = int(state.retry_count or 0)

    if decision in ("pass", "conditional_pass"):
        final_decision = "pass"
        new_retry = retry
    elif decision == "critical_fail":
        final_decision = "pause"
        new_retry = retry
    elif decision == "rollback":
        if retry < MAX_RETRY:
            final_decision = "retry"
            new_retry = retry + 1
        else:
            final_decision = "pause"
            new_retry = retry
    else:
        final_decision = "pass"
        new_retry = retry

    logger.info("critic decision=%s score=%s final=%s retry=%s",
                decision, report.get("overall_score"), final_decision, new_retry)
    return CriticDecisionOutput(
        critic_decision=decision,
        retry_count=new_retry,
        final_decision=final_decision,
    )
