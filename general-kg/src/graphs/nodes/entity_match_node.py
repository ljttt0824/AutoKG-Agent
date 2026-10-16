"""实体匹配节点（Worker 层 W5）

对抽取实体做集合内去重与规范化（文本层匹配），保留代表性实体并维护别名。
"""
import re
from typing import Dict, List
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import EntityMatchInput, EntityMatchOutput


def _normalize(name: str) -> str:
    """规范化名称用于去重匹配"""
    s = name or ""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", s)
    return s


def entity_match_node(
    state: EntityMatchInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> EntityMatchOutput:
    """
    title: 实体匹配去重
    desc: 对抽取实体进行规范化去重，合并同义别名，作为后续定义与图谱构建的权威实体集。
    """
    ctx = runtime.context

    entities = state.entities or []
    seen: Dict[str, dict] = {}
    merged = 0

    for ent in entities:
        if not isinstance(ent, dict):
            continue
        name = ent.get("name")
        if not name:
            continue
        key = _normalize(name)
        if key in seen:
            # 合并 origin forms，保留权重更高者
            existing = seen[key]
            merged += 1
            forms = existing.get("original_forms", [])
            for f in ent.get("original_forms", []) or []:
                if f not in forms:
                    forms.append(f)
            existing["original_forms"] = forms
            if float(ent.get("weight", 0) or 0) > float(existing.get("weight", 0) or 0):
                existing["weight"] = ent.get("weight")
                existing["weight_level"] = ent.get("weight_level")
            existing["occurrences"] = (existing.get("occurrences") or []) + (ent.get("occurrences") or [])
        else:
            ent.setdefault("original_forms", [ent.get("name")])
            ent.setdefault("occurrences", [])
            seen[key] = ent

    matched = list(seen.values())
    stats = {"total_input": len(entities), "merged": merged, "unique": len(matched)}
    return EntityMatchOutput(matched_entities=matched, match_stats=stats)
