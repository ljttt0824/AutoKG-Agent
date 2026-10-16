"""图谱构建节点（Worker 层 W10）

整合实体/关系定义与类型标准化结果，组装最终知识图谱并计算图指标。

质量增强策略：
1) 关系类型白名单对齐：将 LLM 抽取的关系类型就近映射到 schema 白名单，
   并把实际用到的关系类型回写进 schema.relation_types，保证 schema 与数据一致；
2) 节点属性/溯源增强：依据 schema.node_attribute_schema 对必填业务属性做兜底补全，
   并确保 source_sections 溯源存在，避免大量空属性导致 definition_quality 偏低。
"""
import uuid
import datetime
import re
from typing import Dict, List
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import GraphBuildInput, GraphBuildOutput


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", (name or "").strip().lower())


def _clean_id(value: object, fallback: str) -> str:
    """清洗实体/关系 ID，去除空白与非法字符，避免脏ID破坏图结构。"""
    s = str(value or "").strip()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s or fallback


def _alias_relation_type(rel_type: str, whitelist: List[str]) -> str:
    """将 LLM 抽取的关系类型就近映射到 schema 白名单。

    依次尝试：完全命中 -> 规范化后完全命中 -> 白名单中存在包含关系 ->
    与白名单项编辑距离较近（同义规范化）。均不匹配则保留原类型。
    """
    raw = (rel_type or "").strip()
    if not raw:
        return "related_to"
    norm_raw = _norm(raw)
    if not whitelist:
        return raw
    norm_white = [_norm(w) for w in whitelist]
    # 完全命中
    if raw in whitelist or norm_raw in norm_white:
        return raw if raw in whitelist else whitelist[norm_white.index(norm_raw)]
    # 包含关系（单向：白名单项包含原类型 或 原类型包含白名单项）
    for i, wn in enumerate(norm_white):
        if wn and (wn in norm_raw or norm_raw in wn):
            return whitelist[i]
    # 编辑距离兜底（同义别名：如 来源于/来源自/来源）
    def _norm_alias(s: str) -> str:
        table = {"来源于": "来源", "来源自": "来源", "来源": "来源",
                 "属于": "属于", "包含": "包含", "依赖于": "依赖", "依赖": "依赖"}
        a = _norm(s)
        return table.get(a, a)
    a_raw = _norm_alias(raw)
    for i, w in enumerate(whitelist):
        if _norm_alias(w) and _norm_alias(w) == a_raw:
            return whitelist[i]
    return raw


def _compute_metrics(nodes: List[dict], edges: List[dict]) -> dict:
    """计算基础图指标（连通分量、平均度、密度、孤立率等）"""
    n = len(nodes)
    e = len(edges)
    degree: Dict[str, int] = {}
    for node in nodes:
        degree[node["id"]] = 0
    for edge in edges:
        degree[edge["source"]] = degree.get(edge["source"], 0) + 1
        degree[edge["target"]] = degree.get(edge["target"], 0) + 1

    # 并查集求连通分量
    parent = {node["id"]: node["id"] for node in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for edge in edges:
        if edge["source"] in parent and edge["target"] in parent:
            union(edge["source"], edge["target"])

    comp_map: Dict[str, int] = {}
    for nid in parent:
        root = find(nid)
        comp_map[root] = comp_map.get(root, 0) + 1
    connected_components = len(comp_map)
    largest_ratio = (max(comp_map.values()) / n) if n > 0 and comp_map else 0.0

    avg_degree = (2 * e / n) if n > 0 else 0.0
    density = (2 * e / (n * (n - 1))) if n > 1 else 0.0
    isolated = sum(1 for d in degree.values() if d == 0)

    return {
        "node_count": n,
        "edge_count": e,
        "average_degree": round(avg_degree, 4),
        "connected_components": connected_components,
        "density": round(density, 4),
        "largest_component_ratio": round(largest_ratio, 4),
        "isolated_entity_count": isolated,
    }


def graph_build_node(
    state: GraphBuildInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> GraphBuildOutput:
    """
    title: 图谱构建
    desc: 整合实体/关系定义与类型标准化结果，组装最终知识图谱 JSON，并做关系类型白名单对齐与属性/溯源补全。
    """
    ctx = runtime.context

    entities = state.entity_definitions or []
    relations = state.relation_definitions or []
    typestd = state.type_standardization or {}
    task_contract = state.task_contract or {}
    execution_plan = state.execution_plan or {}
    domain_profile = execution_plan.get("domain_profile") or {}

    # 领域自适应 schema：优先取自执行计划领域本体，回退到内部标准类型
    _dp_node_types = domain_profile.get("node_types") or []
    _dp_rel_types = domain_profile.get("relation_types") or []
    dp_entity_types = [
        (t.get("type_name"), t.get("description", ""))
        for t in _dp_node_types if isinstance(t, dict) and t.get("type_name")
    ]
    if not dp_entity_types:
        dp_entity_types = [
            ("concept", "核心概念/知识点"),
            ("protocol", "协议/标准"),
            ("technology", "技术/实现"),
            ("physical_media", "物理介质/硬件"),
            ("other", "其他领域实体"),
        ]
    dp_relation_types = [
        (t.get("type_name"), t.get("business_semantics") or t.get("description", ""))
        for t in _dp_rel_types if isinstance(t, dict) and t.get("type_name")
    ]
    if not dp_relation_types:
        dp_relation_types = [
            ("is_a", "is-a 关系"),
            ("part_of", "部分-整体关系"),
            ("depends_on", "依赖/前置关系"),
            ("related_to", "一般关联"),
        ]
    node_attribute_schema = domain_profile.get("node_attribute_schema") or {}

    # 类型映射
    emap: Dict[str, str] = {}
    if isinstance(typestd.get("entity_type_mappings"), list):
        for m in typestd["entity_type_mappings"]:
            if isinstance(m, dict) and m.get("entity_id"):
                emap[_clean_id(m["entity_id"], "")] = m.get("mapped_type") or "unknown"
    rmap: Dict[str, str] = {}
    if isinstance(typestd.get("relation_type_mappings"), list):
        for m in typestd["relation_type_mappings"]:
            if isinstance(m, dict) and m.get("relation_id"):
                rmap[_clean_id(m["relation_id"], "")] = m.get("mapped_type") or "related_to"

    name_to_id: Dict[str, str] = {}
    for e in entities:
        if e.get("name") and e.get("entity_id"):
            name_to_id[_norm(e["name"])] = _clean_id(e["entity_id"], "")

    # 关系类型白名单（来自领域本体 relation_types）
    rel_whitelist = [t for t, _ in dp_relation_types]

    # 节点
    nodes: List[dict] = []
    for e in entities:
        eid = _clean_id(e.get("entity_id"), str(uuid.uuid4()))
        occurrences = e.get("occurrences") or []
        source_sections = [o.get("section", "") for o in occurrences if isinstance(o, dict)]
        # 溯源兜底：未从 occurrences 拿到时，尝试取运行时提供的来源信息
        if not source_sections:
            ss = e.get("source_sections") or []
            if isinstance(ss, list):
                source_sections = [str(s) for s in ss if str(s).strip()]

        # 属性补强：仅映射实体中真实存在的业务字段到 schema 键，缺失的必填属性
        # 记录为空串但**不**用 name/definition 硬填（避免无意义膨胀导致 Critic 上下文超限）。
        attributes: Dict[str, object] = {
            "weight": float(e.get("weight", 0) or 0),
            "occurrence_count": len(occurrences),
        }
        # 溯源字段优先登记（真实值，非占位）
        if source_sections and "来源文档位置" in node_attribute_schema:
            attributes["来源文档位置"] = "；".join(source_sections[:5])
        if source_sections and "来源文档页码" in node_attribute_schema:
            attributes["来源文档页码"] = "；".join(source_sections[:5])
        # 从实体中提取与 schema 键同名/近义的真实业务字段（如 description/职责描述/能力说明等）
        for attr_key in node_attribute_schema:
            if str(attr_key) in attributes:
                continue
            val = e.get(str(attr_key))
            if val is None or val == "":
                continue
            if isinstance(val, (dict, list)):
                attributes[str(attr_key)] = val
            else:
                attributes[str(attr_key)] = str(val)
        # 缺失的 schema 必填属性给出空串占位（保留 schema 契约可见，但不引入 name 冗余）
        for attr_key in node_attribute_schema:
            if str(attr_key) not in attributes:
                attributes[str(attr_key)] = ""

        nodes.append({
            "id": eid,
            "name": e.get("name", ""),
            "type": emap.get(eid, e.get("type", "unknown")),
            "definition": e.get("definition", ""),
            "importance": e.get("importance", 3),
            "original_forms": e.get("original_forms", []),
            "source_sections": source_sections,
            "attributes": attributes,
        })
    # 硬保护：实体资格判定过滤——剔除命中 entity_boundary.excluded_phrases 的不适格节点
    # (如电商中录取通知书/现金/银行卡/档案等"非卖品/材料"不具备商品节点资格)
    _eb = domain_profile.get("entity_boundary") or {}
    excluded_phrases: List[str] = []
    for _p in (_eb.get("excluded_phrases") or []):
        if isinstance(_p, str) and _p.strip():
            excluded_phrases.append(_p.strip())
    for _cls in (_eb.get("excluded_classes") or []):
        if isinstance(_cls, dict):
            for _e in (_cls.get("examples") or []):
                if isinstance(_e, str) and _e.strip():
                    excluded_phrases.append(_e.strip())
    excluded_phrases = [p for p in dict.fromkeys(excluded_phrases) if p]

    def _excluded(name: str) -> bool:
        n = (name or "").lower()
        for p in excluded_phrases:
            if p and p.lower() in n:
                return True
        return False

    kept_nodes: List[dict] = []
    for n in nodes:
        if n["name"] and excluded_phrases and _excluded(n["name"]):
            continue  # 命中明确的非卖品/材料排除词，丢弃
        kept_nodes.append(n)
    nodes = kept_nodes
    node_ids = {n["id"] for n in nodes}

    # 边：关系类型白名单对齐
    edges: List[dict] = []
    used_rel_types: set = set()
    for r in relations:
        rid = _clean_id(r.get("relation_id"), str(uuid.uuid4()))
        src = _clean_id(r.get("source_entity_id"), name_to_id.get(_norm(r.get("source_name")), ""))
        tgt = _clean_id(r.get("target_entity_id"), name_to_id.get(_norm(r.get("target_name")), ""))
        if src not in node_ids:
            src = name_to_id.get(_norm(r.get("source_name")), "")
        if tgt not in node_ids:
            tgt = name_to_id.get(_norm(r.get("target_name")), "")
        if not src or not tgt or src == tgt:
            continue
        # 类型标准化映射 -> 白名单对齐
        raw_type = rmap.get(rid, r.get("relation_type", "related_to"))
        rel_type = _alias_relation_type(str(raw_type), rel_whitelist)
        if rel_type == str(raw_type) and str(raw_type) not in rel_whitelist:
            # 仍不在白名单：合法但未定义的类型，登记进 used 集合以便回写 schema
            rel_type = str(raw_type)
        used_rel_types.add(rel_type)
        edges.append({
            "id": rid,
            "source": src,
            "target": tgt,
            "type": rel_type,
            "definition": r.get("definition", ""),
            "strength": float(r.get("strength", 0) or 0),
            "source_section": r.get("section_id", ""),
            "provenance": {
                "supporting_text": r.get("supporting_text", ""),
                "extraction_method": r.get("extraction_method", "explicit"),
            },
        })

    # schema 与实际数据一致化：确保实际用到的关系类型也登记在 schema.relation_types
    used_descs = {t: d for t, d in dp_relation_types}
    for ut in used_rel_types:
        if ut not in used_descs:
            used_descs[ut] = _alias_relation_type(ut, rel_whitelist)
    final_rel_types = [(t, used_descs.get(t, d)) for t, d in dp_relation_types]
    for ut in used_rel_types:
        if ut not in {t for t, _ in final_rel_types}:
            final_rel_types.append((ut, used_descs.get(ut, ut)))

    metrics = _compute_metrics(nodes, edges)

    knowledge_graph = {
        "graph_id": str(uuid.uuid4()),
        "task_id": str(task_contract.get("task_id", "")),
        "domain": str((task_contract.get("domain") or {}).get("primary", "") or ""),
        "version": "1.0",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metadata": {"total_characters_processed": 0},
        "schema": {
            "entity_types": [{"type_name": t, "description": d, "is_standard": False} for t, d in dp_entity_types],
            "relation_types": [{"type_name": t, "description": d, "is_standard": False} for t, d in final_rel_types],
            "node_attribute_schema": node_attribute_schema,
        },
        "nodes": nodes,
        "edges": edges,
        "source_links": [],
        "graph_metrics": metrics,
        "quality_summary": {},
    }
    return GraphBuildOutput(knowledge_graph=knowledge_graph)
