"""设计确认门节点

在 design_review 之后判断：若 Planner 判定信息不足或设计需要用户确认，
则主图在此暂停返回追问澄清项，等待用户补充后再构建；否则放行进入构图。
"""
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import DesignGateInput, DesignGateOutput


def design_gate_node(
    state: DesignGateInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> DesignGateOutput:
    """
    title: 设计确认门
    desc: 当任务信息不足或设计方案需用户确认时暂停返回追问项；信息充足且已确认则放行构图。
    """
    ctx = runtime.context

    proposal = state.design_proposal or {}
    task_contract = state.task_contract or {}
    clarity = task_contract.get("clarity") or {}
    has_files = bool(state.files)

    # 是否判定需追问
    needs_confirm = bool(
        proposal.get("needs_user_confirmation")
        or clarity.get("needs_human_confirmation")
        or clarity.get("info_sufficiency_level") == "low"
    )
    # 即使标记需确认，若已提供源文档且用户已给出明确领域，可视为基本可构建
    # （仍以 needs_user_confirmation 为准，避免静默跳过用户期望的确认）
    confirm_items = proposal.get("confirmation_items") or clarity.get("clarification_items") or []

    if needs_confirm and confirm_items:
        questions = []
        for i in confirm_items:
            if isinstance(i, dict):
                q = str(i.get("question", ""))
                if q:
                    questions.append({"field": str(i.get("field", "")), "question": q})
        return DesignGateOutput(
            gate_decision="need_confirmation",
            gate_pending=True,
            confirmation_questions=questions,
            gate_reason=f"信息不足或设计待确认，暂不构图，向用户追问 {len(questions)} 项："
                        + "；".join(str(i.get('question', '')) for i in questions),
        )

    return DesignGateOutput(gate_decision="proceed", gate_pending=False, gate_reason="领域/任务信息明确，设计方案已确认，放行构图")
