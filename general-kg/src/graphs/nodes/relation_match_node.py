"""关系匹配节点（Worker 层 W6）

对抽取关系做：实体ID对齐（依据 W5 匹配结果）、自环剔除、方向无关签名去重。
"""
import re
from typing import Dict, List
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import RelationMatchInput, RelationMatchOutput


def _norm(name: str) -> str:
    s = name or ""
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", s.strip().lower())


def relation_match_node(
    state: RelationMatchInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> RelationMatchOutput:
    """
    title: 关系匹配去重
    desc: 依实体匹配结果对齐关系指向的实体 ID，剔除自环并按无方向签名去重关系。
    """
    ctx = runtime.context

    # 建立 规范化名 -> 实体 映射（取匹配后权威实体）
    name_to_ent: Dict[str, dict] = {}
    for ent in state.matched_entities or []:
        if not isinstance(ent, dict):
            continue
        name = ent.get("name")
        if name:
            name_to_ent[_norm(name)] = ent
        for f in ent.get("original_forms", []) or []:
            name_to_ent[_norm(f)] = ent

    relations = state.relations or []
    cleaned: List[dict] = []
    seen_sig: Dict[str, dict] = {}

    for rel in relations:
        if not isinstance(rel, dict):
            continue
        src_name = rel.get("source_name")
        tgt_name = rel.get("target_name")
        if not src_name or not tgt_name:
            continue

        src_ent = name_to_ent.get(_norm(src_name)) or {}
        tgt_ent = name_to_ent.get(_norm(tgt_name)) or {}
        src_id = src_ent.get("entity_id") or rel.get("source_entity_id")
        tgt_id = tgt_ent.get("entity_id") or rel.get("target_entity_id")

        # 自环剔除
        if src_id and tgt_id and src_id == tgt_id:
            continue

        rel["source_entity_id"] = src_id
        rel["target_entity_id"] = tgt_id
        rel["source_name"] = src_ent.get("name") or src_name
        rel["target_name"] = tgt_ent.get("name") or tgt_name

        # 方向无关签名去重
        sig = "|".join(sorted([str(src_id or src_name), str(tgt_id or tgt_name)])) + "|" + str(rel.get("relation_type", ""))
        exist = seen_sig.get(sig)
        if exist is not None:
            # 保留置信度更高者，合并支撑文本
            if float(rel.get("confidence", 0) or 0) > float(exist.get("confidence", 0) or 0):
                seen_sig[sig] = rel
            else:
                exist["supporting_text"] = (exist.get("supporting_text") or "") + "\n" + (rel.get("supporting_text") or "")
        else:
            seen_sig[sig] = rel

    cleaned = list(seen_sig.values())

    # 稳定排序：类型 -> source -> 置信度降序
    cleaned.sort(key=lambda r: (str(r.get("relation_type", "")), str(r.get("source_name", "")), -float(r.get("confidence", 0) or 0)))
    return RelationMatchOutput(matched_relations=cleaned)
