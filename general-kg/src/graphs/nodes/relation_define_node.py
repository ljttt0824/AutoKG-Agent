"""关系定义节点（Worker 层 W8）

为每条关系生成遵循标准模板的定义并评估关系强度，并将 LLM 结果
**合并**回 W6 去重定向后的关系记录，保留 source/target/type/provenance
/溯源等字段，避免赋值被覆盖。
"""
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import RelationDefineInput, RelationDefineOutput
from utils.llm_helper import invoke_llm, extract_json


def _norm(name: str) -> str:
    if name is None:
        return ""
    return "".join(str(name).strip().lower().split()).replace("，", "").replace(",", "")


def relation_define_node(
    state: RelationDefineInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> RelationDefineOutput:
    """
    title: 关系定义
    desc: 依据七种标准关系模板为每条关系生成定义并评估强度，同时保留 W6 匹配阶段的源头/目标/溯源等字段。
    integrations: 大语言模型
    """
    ctx = runtime.context

    matched = state.matched_relations or []
    llm_rels: list = []
    if matched:
        params = {"matched_relations": matched}
        resp_text = invoke_llm(runtime, config, render_params=params)
        parsed = extract_json(resp_text)
        if isinstance(parsed.get("relations"), list):
            llm_rels = parsed["relations"]

    # 构造 LLM 关系映射：优先按 relation_id，其次按 (source_name, target_name, type)
    def_by_id: dict = {}
    def_by_key: dict = {}
    for lr in llm_rels:
        if not isinstance(lr, dict):
            continue
        rid = str(lr.get("relation_id") or lr.get("id") or "").strip()
        key = (
            _norm(lr.get("source_name")),
            _norm(lr.get("target_name")),
            _norm(lr.get("relation_type")),
        )
        if rid:
            def_by_id[rid] = lr
        if key[0] and key[1] and key[2]:
            def_by_key[key] = lr

    merged: list = []
    for mr in matched:
        base = dict(mr)  # 保留 source/target/type/provenance/source_section 等
        rid = str(base.get("relation_id") or base.get("id") or "").strip()
        key = (
            _norm(base.get("source_name")),
            _norm(base.get("target_name")),
            _norm(base.get("relation_type")),
        )
        lr = def_by_id.get(rid) or def_by_key.get(key) or {}
        if isinstance(lr, dict):
            for k in ("definition", "strength", "strength_breakdown"):
                if k in lr and lr.get(k) is not None:
                    base[k] = lr[k]
        merged.append(base)

    return RelationDefineOutput(relation_definitions=merged)
