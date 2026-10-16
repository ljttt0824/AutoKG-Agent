"""实体定义节点（Worker 层 W7）

为每个权威实体生成 2-4 句定义与重要性评分，并将 LLM 结果**合并**回
W5 匹配下的富属性实体，保留 original_forms / occurrences / weight / type
/ role / source_sections 等溯源与权重信息，避免赋值被覆盖。
"""
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import EntityDefineInput, EntityDefineOutput
from utils.llm_helper import invoke_llm, extract_json


def _norm(name: str) -> str:
    """规范化实体名用于匹配：去空白、统一小写、去常见标点。"""
    if name is None:
        return ""
    return "".join(str(name).strip().lower().split()).replace("，", "").replace(",", "")


def entity_define_node(
    state: EntityDefineInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> EntityDefineOutput:
    """
    title: 实体定义
    desc: 结合上下文为每个实体生成简洁定义并评估重要性（1-5 级），同时保留 W5 匹配阶段的富属性（溯源/权重/原始形态等）。
    integrations: 大语言模型
    """
    ctx = runtime.context

    matched = state.matched_entities or []
    llm_entities: list = []
    if matched:
        params = {
            "matched_entities": matched,
            "text_content": (state.text_content or "")[:60000],
        }
        resp_text = invoke_llm(runtime, config, render_params=params)
        parsed = extract_json(resp_text)
        if isinstance(parsed.get("entities"), list):
            llm_entities = parsed["entities"]

    # 构造 LLM 定义映射：优先按 entity_id，其次按规范化名称
    def_by_id: dict = {}
    def_by_name: dict = {}
    for le in llm_entities:
        if not isinstance(le, dict):
            continue
        eid = str(le.get("entity_id") or le.get("id") or "").strip()
        nm = _norm(le.get("name"))
        if not eid and not nm:
            continue
        if eid:
            def_by_id[eid] = le
        if nm:
            def_by_name[nm] = le

    # 合并：以匹配实体为权威基底，仅补充定义/重要性，保留所有富属性
    merged: list = []
    for me in matched:
        base = dict(me)  # 保留 original_forms/occurrences/weight/type/role 等
        eid = str(base.get("entity_id") or base.get("id") or "").strip()
        le = def_by_id.get(eid) or def_by_name.get(_norm(base.get("name"))) or {}
        if isinstance(le, dict):
            for k in ("definition", "importance", "importance_score", "definition_source"):
                if k in le and le.get(k) is not None:
                    base[k] = le[k]
            # 若 LLM 返回了更明确的 type，用于类型修正
            if not base.get("type") and le.get("type"):
                base["type"] = le["type"]
        merged.append(base)

    # 补充 LLM 中新增的权威实体（如匹配阶段遗漏但上下文明确的）
    seen = set()
    for m in merged:
        seen.add(str(m.get("entity_id") or "").strip() or _norm(m.get("name")))
    for le in llm_entities:
        if not isinstance(le, dict):
            continue
        key = str(le.get("entity_id") or "").strip() or _norm(le.get("name"))
        if key and key not in seen:
            merged.append(dict(le))
            seen.add(key)

    return EntityDefineOutput(entity_definitions=merged)
