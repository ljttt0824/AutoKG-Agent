"""
Quality assessor for evaluating the quality of the knowledge graph.
"""

import logging
import json
import os
import math
from typing import Dict, List, Any, Tuple
import re

import config
from utils.llm_utils import LLMClient

logger = logging.getLogger(__name__)

class QualityAssessor:
    """
    Class for assessing the quality of the knowledge graph.
    """

    def __init__(self, model_name: str = config.DEFAULT_MODEL, api_key: str = None, api_base: str = None):
        """
        Initialize the quality assessor.

        Args:
            model_name: Name of the LLM model to use for quality assessment
            api_key: API key for the LLM service
            api_base: API base URL for the LLM service
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.llm_client = LLMClient(model_name=model_name, api_key=api_key, api_base=api_base)
        logger.info(f"Initialized quality assessor with model {model_name}")

    def assess(self, knowledge_graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the quality of the knowledge graph.

        Args:
            knowledge_graph: Dictionary containing the knowledge graph

        Returns:
            Dictionary containing quality assessment results
        """
        logger.info("Assessing knowledge graph quality")

        entities = knowledge_graph.get("entities", [])
        relationships = knowledge_graph.get("relationships", [])

        # 计算没有参与任何关系的实体数量
        entity_ids = {entity.get("id") for entity in entities}
        entity_in_relationships = set()

        for rel in relationships:
            source_id = rel.get("source")
            target_id = rel.get("target")
            if source_id:
                entity_in_relationships.add(source_id)
            if target_id:
                entity_in_relationships.add(target_id)

        # 没有参与任何关系的实体
        entities_without_relationships = entity_ids - entity_in_relationships
        num_entities_without_relationships = len(entities_without_relationships)

        # 计算没有关系实体的百分比
        entities_without_relationships_percentage = 0
        if entity_ids:
            entities_without_relationships_percentage = (num_entities_without_relationships / len(entity_ids)) * 100

        logger.info(f"找到 {num_entities_without_relationships} 个没有参与任何关系的实体，占比 {entities_without_relationships_percentage:.2f}%")

        # 生成必要的分析数据，而不是从knowledge_graph中获取
        analysis = self._generate_analysis(entities, relationships)

        # Calculate quality metrics
        entity_coverage = self._calculate_entity_coverage(entities)
        relationship_coverage = self._calculate_relationship_coverage(relationships)

        # 使用生成的分析数据
        entity_type_balance = self._calculate_entity_type_balance(analysis["entity_type_distribution"])
        relationship_type_balance = self._calculate_relationship_type_balance(analysis["relationship_type_distribution"])
        graph_connectivity = self._calculate_graph_connectivity(analysis)
        chapter_coverage = self._calculate_chapter_coverage(analysis["chapter_distribution"])

        # Compare with manual knowledge graph
        manual_comparison = self._compare_with_manual_kg(entities, relationships)

        # 计算新增的基于LLM的质量指标
        logger.info("计算LLM质量指标: 平均置信度、平均清晰度、平均相关性")
        # 为了节省API调用，选择一部分关系进行评估
        sample_size = min(50, len(relationships))  # 最多评估50个关系
        if sample_size > 0:
            import random
            sampled_relationships = random.sample(relationships, sample_size)
            llm_metrics = self._calculate_llm_quality_metrics(entities, sampled_relationships)
        else:
            llm_metrics = {
                "average_confidence": {"score": 0},
                "average_clarity": {"score": 0},
                "average_relevance": {"score": 0}
            }

        # Calculate overall quality score (包含新指标)
        overall_score = self._calculate_overall_score(
            entity_coverage,
            relationship_coverage,
            entity_type_balance,
            relationship_type_balance,
            graph_connectivity,
            chapter_coverage,
            manual_comparison,
            llm_metrics
        )

        return {
            "entity_coverage": entity_coverage,
            "relationship_coverage": relationship_coverage,
            "entity_type_balance": entity_type_balance,
            "relationship_type_balance": relationship_type_balance,
            "graph_connectivity": graph_connectivity,
            "chapter_coverage": chapter_coverage,
            "manual_comparison": manual_comparison,
            "llm_quality_metrics": llm_metrics,
            "overall_score": overall_score,
            "entities_without_relationships": {
                "count": num_entities_without_relationships,
                "percentage": round(entities_without_relationships_percentage, 2)
            }
        }

    def _generate_analysis(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate analysis data from entities and relationships.

        Args:
            entities: List of entities
            relationships: List of relationships

        Returns:
            Dictionary containing analysis data
        """
        from collections import Counter, defaultdict

        # 计算实体类型分布
        entity_types = [entity.get("type", "unknown") for entity in entities]
        entity_type_distribution = dict(Counter(entity_types))

        # 计算关系类型分布
        relationship_types = [rel.get("type", "unknown") for rel in relationships]
        relationship_type_distribution = dict(Counter(relationship_types))

        # 计算章节分布
        chapters = []
        for entity in entities:
            entity_chapter = entity.get("chapter", "")
            if entity_chapter:
                chapters.append(entity_chapter)

        chapter_distribution = dict(Counter(chapters))

        # 以名称为基础建立实体索引
        entity_name_to_index = {}
        for i, entity in enumerate(entities):
            entity_name = entity.get("name", "").lower()
            if entity_name:
                # 可能有多个实体具有相同名称，保留第一个
                if entity_name not in entity_name_to_index:
                    entity_name_to_index[entity_name] = i

        logger.info(f"Indexed {len(entity_name_to_index)} unique entity names")

        # 基于 source_name 和 target_name 构建邻接表
        adjacency_list = defaultdict(list)
        connected_entities = set()  # 跟踪已连接的实体

        for rel in relationships:
            source_name = rel.get("source_name", "").lower()
            target_name = rel.get("target_name", "").lower()

            # 验证源实体和目标实体名称存在
            if source_name and target_name:
                source_idx = entity_name_to_index.get(source_name)
                target_idx = entity_name_to_index.get(target_name)

                if source_idx is not None and target_idx is not None:
                    # 添加到邻接表
                    adjacency_list[source_idx].append(target_idx)
                    # 对于无向图分析，也添加反向连接
                    adjacency_list[target_idx].append(source_idx)

                    # 记录连接的实体
                    connected_entities.add(source_idx)
                    connected_entities.add(target_idx)

                    logger.debug(f"建立连接: {source_name} ({source_idx}) -> {target_name} ({target_idx})")
                else:
                    if source_idx is None:
                        logger.warning(f"未找到源实体名称 '{source_name}' 的索引")
                    if target_idx is None:
                        logger.warning(f"未找到目标实体名称 '{target_name}' 的索引")

        # 计算连接的实体数和孤立的实体数
        num_connected_entities = len(connected_entities)
        num_isolated_entities = len(entities) - num_connected_entities

        logger.info(f"Total entities: {len(entities)}")
        logger.info(f"Connected entities: {num_connected_entities}")
        logger.info(f"Isolated entities: {num_isolated_entities}")
        logger.info(f"Total relationships: {len(relationships)}")
        logger.info(f"Relationships used in adjacency list: {sum(len(neighbors) for neighbors in adjacency_list.values())//2}")

        # 返回分析结果
        return {
            "entity_type_distribution": entity_type_distribution,
            "relationship_type_distribution": relationship_type_distribution,
            "chapter_distribution": chapter_distribution,
            "adjacency_list": dict(adjacency_list),
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "connected_entity_count": num_connected_entities,
            "isolated_entity_count": num_isolated_entities
        }

    def _compare_with_manual_kg(self, auto_entities: List[Dict[str, Any]],
                              auto_relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare automatically constructed knowledge graph with manual one.

        Args:
            auto_entities: List of entities from automatic construction
            auto_relationships: List of relationships from automatic construction

        Returns:
            Dictionary containing comparison results
        """
        logger.info("Comparing with manual knowledge graph")

        # Import these here to avoid making them project dependencies
        import difflib
        import re
        from collections import OrderedDict, defaultdict

        # Load manual knowledge graph
        manual_kg_path = os.path.join(config.DATA_DIR, "manual_kg")
        manual_entities_path = os.path.join(manual_kg_path, "entity.json")
        manual_relationships_path = os.path.join(manual_kg_path, "relation.json")

        logger.info(f"Manual KG path: {manual_kg_path}")
        logger.info(f"Manual entities path: {manual_entities_path}")
        logger.info(f"Manual relationships path: {manual_relationships_path}")

        try:
            # Check if the files exist
            if not os.path.exists(manual_entities_path):
                logger.error(f"Manual entities file does not exist: {manual_entities_path}")
                return {
                    "entity_coverage": 0,
                    "relationship_coverage": 0,
                    "error": f"Manual entities file not found: {manual_entities_path}"
                }

            if not os.path.exists(manual_relationships_path):
                logger.error(f"Manual relationships file does not exist: {manual_relationships_path}")
                return {
                    "entity_coverage": 0,
                    "relationship_coverage": 0,
                    "error": f"Manual relationships file not found: {manual_relationships_path}"
                }

            # Load the files
            with open(manual_entities_path, 'r', encoding='utf-8-sig') as f:
                manual_entities_data = json.load(f)
            with open(manual_relationships_path, 'r', encoding='utf-8-sig') as f:
                manual_relationships_data = json.load(f)

            # 处理不同格式的manual_entities_data (可能是dict包含data字段，或直接是list)
            if isinstance(manual_entities_data, dict) and "data" in manual_entities_data:
                manual_entities = manual_entities_data.get("data", [])
            else:
                # 假设直接是实体列表
                manual_entities = manual_entities_data

            # 同样处理manual_relationships_data
            if isinstance(manual_relationships_data, dict) and "data" in manual_relationships_data:
                manual_relationships = manual_relationships_data.get("data", [])
            else:
                # 假设直接是关系列表
                manual_relationships = manual_relationships_data

            logger.info(f"Successfully loaded manual KG with {len(manual_entities)} entities and {len(manual_relationships)} relationships")
        except Exception as e:
            logger.error(f"Error loading manual knowledge graph: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "entity_coverage": 0,
                "relationship_coverage": 0,
                "error": str(e)
            }

        logger.info(f"Processing auto KG with {len(auto_entities)} entities and {len(auto_relationships)} relationships")

        # 预处理函数：稳定地清理、标准化文本
        def normalize_name(name):
            """将实体名称标准化为仅保留字母数字的小写形式"""
            if not name:
                return ""
            # 转小写，移除非字母数字字符
            norm = re.sub(r'[^a-z0-9]', '', name.lower())
            return norm

        # 获取实体的所有可能名称形式 - 确保稳定性
        def get_entity_name_variants(entity):
            """获取实体的所有可能名称变体，并确保结果是有序列表"""
            variants = []

            # 处理实体名称 (格式可能是 "english_name-chinese_name")
            if "name" in entity:
                full_name = entity["name"].lower()
                variants.append(full_name)

                # 分离英文和中文名称
                if "-" in full_name:
                    english_name = full_name.split("-")[0].strip()
                    variants.append(english_name)
                    # 对于像 "transmission control protocol/internet protocol" 这样的名称
                    # 也添加其部分作为变体
                    parts = english_name.split()
                    if len(parts) > 1:
                        variants.append(parts[0])  # 第一个词
                        variants.append(parts[-1])  # 最后一个词

            # 添加英文名（如果存在）
            if "english_name" in entity and entity["english_name"]:
                english = entity["english_name"].lower()
                variants.append(english)

                # 处理英文名中的常见分隔符
                for sep in ['/', '-', ' ']:
                    if sep in english:
                        for part in english.split(sep):
                            if part.strip():
                                variants.append(part.strip())

            # 添加简化名称（如果存在）
            if "simplified_name" in entity and entity["simplified_name"]:
                variants.append(entity["simplified_name"].lower())

            # 应用预处理，生成清理后的变体
            processed_variants = []
            for v in variants:
                # 去除括号内容及括号
                processed = re.sub(r'\([^)]*\)', '', v).strip()
                if processed:
                    processed_variants.append(processed)
                    # 如果处理后的变体包含空格，也添加无空格版本
                    if ' ' in processed:
                        processed_variants.append(processed.replace(' ', ''))

                    # 对于协议名，尝试添加常见缩写
                    if "protocol" in processed:
                        abbr = ''.join(word[0] for word in processed.split() if word != "protocol")
                        if len(abbr) > 1:  # 确保缩写至少有2个字符
                            processed_variants.append(abbr)

            # 合并所有变体，去重并排序以确保稳定性
            all_variants = variants + processed_variants
            unique_variants = []
            for v in all_variants:
                if v and v not in unique_variants:
                    unique_variants.append(v)

            # 按字母顺序排序以确保稳定性
            return sorted(unique_variants)

        # 构建手动实体的名称变体字典 - 使用OrderedDict确保顺序稳定
        manual_entity_variants = OrderedDict()
        manual_entity_name_to_id = {}
        all_manual_variants = []

        # 首先对手动实体进行排序，确保处理顺序稳定
        sorted_manual_entities = sorted(manual_entities, key=lambda e: str(e.get("id", "")))

        for entity in sorted_manual_entities:
            entity_id = entity.get("id", "")
            variants = get_entity_name_variants(entity)
            manual_entity_variants[entity_id] = variants

            # 创建从变体到ID的映射
            for variant in variants:
                if variant not in all_manual_variants:
                    all_manual_variants.append(variant)
                manual_entity_name_to_id[variant] = entity_id

        # 构建自动实体的名称变体字典
        auto_entity_variants = OrderedDict()
        auto_entity_name_to_id = {}
        all_auto_variants = []

        # 同样对自动实体进行排序
        sorted_auto_entities = sorted(auto_entities, key=lambda e: str(e.get("id", "")))

        for entity in sorted_auto_entities:
            entity_id = entity.get("id", "")
            # 为自动实体创建一个假的类似格式的对象
            entity_obj = {
                "name": entity.get("name", ""),
                "english_name": entity.get("name", ""),  # 使用name作为english_name
                "simplified_name": ""
            }
            variants = get_entity_name_variants(entity_obj)
            auto_entity_variants[entity_id] = variants

            # 创建从变体到ID的映射
            for variant in variants:
                if variant not in all_auto_variants:
                    all_auto_variants.append(variant)
                auto_entity_name_to_id[variant] = entity_id

        logger.info(f"Extracted {len(all_manual_variants)} unique entity name variants from manual KG")
        logger.info(f"Extracted {len(all_auto_variants)} unique entity name variants from auto KG")

        # 使用相似度匹配实体 - 确保稳定的匹配结果
        similarity_threshold = 0.95
        entity_matches = OrderedDict()  # manual_id -> auto_id
        entity_match_types = defaultdict(int)  # 记录不同类型的匹配数量

        # 修改匹配逻辑：每个自动实体只匹配到一个最相似的手动实体
        # 首先为每个自动实体找到最佳匹配的手动实体
        auto_to_manual_matches = {}  # auto_id -> (manual_id, score, variant_pair)

        # 对自动实体进行排序以确保结果稳定
        for auto_id, auto_variants in sorted(auto_entity_variants.items(), key=lambda x: x[0]):
            best_manual_id = None
            best_score = 0
            best_variant_pair = None
            match_type = "none"

            # 对于每个自动实体变体，查找最佳匹配的手动实体
            for auto_variant in auto_variants:
                # 精确匹配
                if auto_variant in all_manual_variants:
                    manual_id = manual_entity_name_to_id.get(auto_variant)
                    if manual_id and (best_manual_id is None or best_score < 1.0):
                        best_manual_id = manual_id
                        best_score = 1.0
                        best_variant_pair = (auto_variant, auto_variant)
                        match_type = "exact"
                        break  # 找到精确匹配就停止搜索

                # 如果没有精确匹配，尝试模糊匹配
                if best_score < 1.0:
                    # 按稳定的顺序遍历手动实体ID
                    for manual_id, manual_variants in sorted(manual_entity_variants.items(), key=lambda x: x[0]):
                        for manual_variant in manual_variants:
                            # 使用difflib获取字符串相似度
                            similarity = difflib.SequenceMatcher(None, auto_variant, manual_variant).ratio()
                            if similarity > best_score and similarity >= similarity_threshold:
                                best_manual_id = manual_id
                                best_score = similarity
                                best_variant_pair = (auto_variant, manual_variant)
                                match_type = "fuzzy"

            # 如果找到匹配，保存它
            if best_manual_id and best_score >= similarity_threshold:
                auto_to_manual_matches[auto_id] = (best_manual_id, best_score, best_variant_pair, match_type)

        # 根据得分排序，确保高分匹配优先
        sorted_matches = sorted(auto_to_manual_matches.items(), key=lambda x: (-x[1][1], x[0]))

        # 使用已分配的手动实体集合来跟踪已匹配的手动实体
        assigned_manual_entities = set()

        # 根据排序后的匹配进行分配
        for auto_id, (manual_id, score, variant_pair, match_type) in sorted_matches:
            # 如果这个手动实体已经被分配，跳过
            if manual_id in assigned_manual_entities:
                continue

            # 分配匹配并记录
            entity_matches[manual_id] = (auto_id, score, variant_pair)
            entity_match_types[match_type] += 1
            assigned_manual_entities.add(manual_id)

            # 记录日志
            auto_name = next((e["name"] for e in auto_entities if e.get("id") == auto_id), "Unknown")
            manual_name = next((e["name"] for e in manual_entities if e.get("id") == manual_id), "Unknown")
            logger.info(f"Matched: '{manual_name}' → '{auto_name}' with score {score:.2f} ({match_type})")

        # 统计实体匹配结果
        matched_entity_count = len(entity_matches)
        entity_coverage = matched_entity_count / len(manual_entities) * 100 if manual_entities else 0

        logger.info(f"Matched {matched_entity_count}/{len(manual_entities)} entities ({entity_coverage:.2f}%)")
        if matched_entity_count > 0:
            sample_matches = list(entity_matches.items())[:5]
            for manual_id, (auto_id, score, _) in sample_matches:
                manual_name = next((e["name"] for e in manual_entities if e.get("id") == manual_id), "Unknown")
                auto_name = next((e["name"] for e in auto_entities if e.get("id") == auto_id), "Unknown")
                logger.info(f"Sample match: '{manual_name}' → '{auto_name}' (score: {score:.2f})")

        # 创建ID到名称的映射
        manual_id_to_name = OrderedDict()
        for e in sorted_manual_entities:
            if "id" in e and "name" in e:
                manual_id_to_name[e["id"]] = e["name"].lower()

        auto_id_to_name = OrderedDict()
        for e in sorted_auto_entities:
            if "id" in e and "name" in e:
                auto_id_to_name[e["id"]] = e["name"].lower()

        # 创建实体名称到变体的映射
        manual_name_to_variants = OrderedDict()
        for entity_id, variants in manual_entity_variants.items():
            if entity_id in manual_id_to_name:
                name = manual_id_to_name[entity_id]
                manual_name_to_variants[name] = variants

        # 构建基于名称的实体匹配映射
        manual_to_auto_name_map = OrderedDict()
        for manual_id, (auto_id, _, _) in entity_matches.items():
            if manual_id in manual_id_to_name and auto_id in auto_id_to_name:
                manual_name = manual_id_to_name[manual_id]
                auto_name = auto_id_to_name[auto_id]
                manual_to_auto_name_map[manual_name] = auto_name

        # ======== 关系匹配部分 - 新的稳定算法 ========

        # 构建手动关系对集合（基于实体名称，并确保顺序稳定）
        manual_rel_name_pairs = []

        # 对手动关系进行排序以确保稳定性
        sorted_manual_relationships = sorted(manual_relationships, key=lambda r: f"{r.get('subject', '')}-{r.get('object', '')}")

        for rel in sorted_manual_relationships:
            # 提取英文部分作为实体名称
            subject = rel.get("subject", "").lower()
            object_entity = rel.get("object", "").lower()

            # 提取英文部分 (通常格式为 "english_name-chinese_name")
            if "-" in subject:
                english_subject = subject.split("-")[0].strip()
            else:
                english_subject = subject

            if "-" in object_entity:
                english_object = object_entity.split("-")[0].strip()
            else:
                english_object = object_entity

            # 去除常见括号内容，如 "internet protocol (IP)" -> "internet protocol"
            clean_subject = re.sub(r'\([^)]*\)', '', english_subject).strip()
            clean_object = re.sub(r'\([^)]*\)', '', english_object).strip()

            # 使用元组而不是集合，确保顺序
            rel_pair = (clean_subject, clean_object, rel.get("id", ""))
            if rel_pair not in manual_rel_name_pairs:
                manual_rel_name_pairs.append(rel_pair)

        # 构建自动关系对集合 - 也使用有序列表而非集合
        auto_rel_name_pairs = []

        # 对自动关系进行排序以确保稳定性
        sorted_auto_relationships = sorted(auto_relationships, key=lambda r: f"{r.get('source', r.get('entity1', ''))}-{r.get('target', r.get('entity2', ''))}")

        for rel in sorted_auto_relationships:
            # 兼容性处理：支持 source/target 和 entity1/entity2 两种格式
            source_id = rel.get("source") or rel.get("entity1")
            target_id = rel.get("target") or rel.get("entity2")

            if not source_id or not target_id:
                logger.warning(f"Relationship missing source/target or entity1/entity2: {rel}")
                continue

            if source_id in auto_id_to_name and target_id in auto_id_to_name:
                source_name = auto_id_to_name[source_id]
                target_name = auto_id_to_name[target_id]

                # 清理和标准化名称
                clean_source = re.sub(r'\([^)]*\)', '', source_name).strip().lower()
                clean_target = re.sub(r'\([^)]*\)', '', target_name).strip().lower()

                # 提取英文部分 (如果有中文分隔符)
                if "-" in clean_source:
                    eng_source = clean_source.split("-")[0].strip()
                else:
                    eng_source = clean_source

                if "-" in clean_target:
                    eng_target = clean_target.split("-")[0].strip()
                else:
                    eng_target = clean_target

                # 使用元组创建关系对，包含关系ID以确保唯一性
                rel_pair = (eng_source, eng_target, rel.get("id", ""))
                if rel_pair not in auto_rel_name_pairs:
                    auto_rel_name_pairs.append(rel_pair)

                # 添加反向关系对 - 确保关系ID标记为反向
                rev_rel_pair = (eng_target, eng_source, f"rev_{rel.get('id', '')}")
                if rev_rel_pair not in auto_rel_name_pairs:
                    auto_rel_name_pairs.append(rev_rel_pair)

        # 使用多种稳定策略匹配关系
        matched_rel_pairs = []
        relationship_match_types = defaultdict(int)

        # 创建自动关系的查找字典
        auto_rel_lookup = OrderedDict()
        for src, tgt, rel_id in auto_rel_name_pairs:
            # 使用规范化名称作为键
            norm_src = normalize_name(src)
            norm_tgt = normalize_name(tgt)
            key = (norm_src, norm_tgt)

            if key not in auto_rel_lookup:
                auto_rel_lookup[key] = []
            auto_rel_lookup[key].append((src, tgt, rel_id))

        # 标准化手动关系名称
        manual_rel_norm_pairs = []
        for src, tgt, rel_id in manual_rel_name_pairs:
            norm_src = normalize_name(src)
            norm_tgt = normalize_name(tgt)
            if norm_src and norm_tgt:  # 确保不是空字符串
                manual_rel_norm_pairs.append((src, tgt, norm_src, norm_tgt, rel_id))

        # 使用确定性顺序处理手动关系
        for src, tgt, norm_src, norm_tgt, rel_id in sorted(manual_rel_norm_pairs):
            # 直接匹配
            if (norm_src, norm_tgt) in auto_rel_lookup:
                matched_rel_pairs.append((src, tgt, "direct"))
                relationship_match_types["direct"] += 1
                continue

            # 实体映射匹配 - 使用已匹配的实体
            found = False

            # 提取匹配的实体名称
            for manual_name, auto_name in manual_to_auto_name_map.items():
                manual_variants = manual_name_to_variants.get(manual_name, [])

                # 检查源实体是否匹配
                is_src_match = src in manual_variants

                # 找到匹配的目标实体
                for manual_target, auto_target in manual_to_auto_name_map.items():
                    manual_target_variants = manual_name_to_variants.get(manual_target, [])

                    # 检查目标实体是否匹配
                    is_tgt_match = tgt in manual_target_variants

                    if is_src_match and is_tgt_match:
                        # 提取和规范化自动匹配的名称
                        norm_auto_src = normalize_name(auto_name)
                        norm_auto_tgt = normalize_name(auto_target)

                        # 检查是否存在对应的自动关系
                        if (norm_auto_src, norm_auto_tgt) in auto_rel_lookup:
                            matched_rel_pairs.append((src, tgt, "entity_mapped"))
                            relationship_match_types["entity_mapped"] += 1
                            found = True
                            break

                        # 检查反向关系
                        if (norm_auto_tgt, norm_auto_src) in auto_rel_lookup:
                            matched_rel_pairs.append((src, tgt, "entity_mapped_reversed"))
                            relationship_match_types["entity_mapped_reversed"] += 1
                            found = True
                            break

                if found:
                    break

        # 统计关系匹配结果
        matched_relationship_count = len(matched_rel_pairs)
        relationship_coverage = matched_relationship_count / len(manual_relationships) * 100 if manual_relationships else 0

        logger.info(f"Found {len(manual_rel_name_pairs)} relation pairs in manual KG")
        logger.info(f"Found {len(auto_rel_name_pairs)} relation pairs in auto KG")
        logger.info(f"Matched {matched_relationship_count}/{len(manual_relationships)} relationships ({relationship_coverage:.2f}%)")

        # 记录匹配方法的分布
        for match_type, count in relationship_match_types.items():
            logger.info(f"  - {match_type}: {count} matches")

        # 记录一些示例匹配
        if matched_rel_pairs:
            sample_matches = matched_rel_pairs[:5]
            logger.info("Sample relationship matches:")
            for src, tgt, match_type in sample_matches:
                logger.info(f"  - '{src}' → '{tgt}' (matched by: {match_type})")

        return {
            "entity_coverage": entity_coverage,
            "relationship_coverage": relationship_coverage,
            "manual_entity_count": len(manual_entities),
            "manual_relationship_count": len(manual_relationships),
            "matched_entity_count": matched_entity_count,
            "matched_relationship_count": matched_relationship_count,
            "entity_match_types": dict(entity_match_types),
            "relationship_match_types": dict(relationship_match_types)
        }

    def _calculate_entity_coverage(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate entity coverage metrics.

        Args:
            entities: List of entities

        Returns:
            Dictionary containing entity coverage metrics
        """
        total_entities = len(entities)

        # Count entities with definitions
        entities_with_definition = sum(1 for entity in entities if entity.get("definition"))

        # Count entities with verified occurrences
        entities_verified = sum(1 for entity in entities if entity.get("verified", True))

        # Calculate percentages
        definition_percentage = entities_with_definition / total_entities * 100 if total_entities > 0 else 0
        verified_percentage = entities_verified / total_entities * 100 if total_entities > 0 else 0

        # Calculate score (0-100)
        score = (definition_percentage + verified_percentage) / 2

        return {
            "total_entities": total_entities,
            "entities_with_definition": entities_with_definition,
            "entities_verified": entities_verified,
            "definition_percentage": definition_percentage,
            "verified_percentage": verified_percentage,
            "score": score
        }

    def _calculate_relationship_coverage(self, relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate relationship coverage metrics.

        Args:
            relationships: List of relationships

        Returns:
            Dictionary containing relationship coverage metrics
        """
        total_relationships = len(relationships)

        # Count relationships with definitions
        relationships_with_definition = sum(1 for rel in relationships if rel.get("definition"))

        # Count relationships with verified occurrences
        relationships_verified = sum(1 for rel in relationships if rel.get("verified", True))

        # Calculate percentages
        definition_percentage = relationships_with_definition / total_relationships * 100 if total_relationships > 0 else 0
        verified_percentage = relationships_verified / total_relationships * 100 if total_relationships > 0 else 0

        # Calculate score (0-100)
        score = (definition_percentage + verified_percentage) / 2

        return {
            "total_relationships": total_relationships,
            "relationships_with_definition": relationships_with_definition,
            "relationships_verified": relationships_verified,
            "definition_percentage": definition_percentage,
            "verified_percentage": verified_percentage,
            "score": score
        }

    def _calculate_entity_type_balance(self, entity_type_distribution: Dict[str, int]) -> Dict[str, Any]:
        """
        Calculate entity type balance metrics.

        Args:
            entity_type_distribution: Distribution of entity types

        Returns:
            Dictionary containing entity type balance metrics
        """
        total_entities = sum(entity_type_distribution.values())

        # Calculate percentages for each type
        type_percentages = {
            entity_type: count / total_entities * 100 if total_entities > 0 else 0
            for entity_type, count in entity_type_distribution.items()
        }

        # Calculate balance score
        # A perfectly balanced distribution would have equal percentages for all types
        # We calculate how far the actual distribution is from the ideal
        num_types = len(entity_type_distribution)
        ideal_percentage = 100 / num_types if num_types > 0 else 0

        # Calculate the sum of absolute differences from the ideal percentage
        sum_diff = sum(abs(percentage - ideal_percentage) for percentage in type_percentages.values())

        # Normalize to a 0-100 score (0 = completely unbalanced, 100 = perfectly balanced)
        # The maximum possible sum of differences is 2 * (100 - ideal_percentage)
        max_diff = 2 * (100 - ideal_percentage) if num_types > 1 else 0
        balance_score = 100 - (sum_diff / max_diff * 100) if max_diff > 0 else 100

        return {
            "type_distribution": entity_type_distribution,
            "type_percentages": type_percentages,
            "balance_score": balance_score
        }

    def _calculate_relationship_type_balance(self, relationship_type_distribution: Dict[str, int]) -> Dict[str, Any]:
        """
        Calculate relationship type balance metrics.

        Args:
            relationship_type_distribution: Distribution of relationship types

        Returns:
            Dictionary containing relationship type balance metrics
        """
        total_relationships = sum(relationship_type_distribution.values())

        # Calculate percentages for each type
        type_percentages = {
            relationship_type: count / total_relationships * 100 if total_relationships > 0 else 0
            for relationship_type, count in relationship_type_distribution.items()
        }

        # Calculate balance score
        # A perfectly balanced distribution would have equal percentages for all types
        # We calculate how far the actual distribution is from the ideal
        num_types = len(relationship_type_distribution)
        ideal_percentage = 100 / num_types if num_types > 0 else 0

        # Calculate the sum of absolute differences from the ideal percentage
        sum_diff = sum(abs(percentage - ideal_percentage) for percentage in type_percentages.values())

        # Normalize to a 0-100 score (0 = completely unbalanced, 100 = perfectly balanced)
        # The maximum possible sum of differences is 2 * (100 - ideal_percentage)
        max_diff = 2 * (100 - ideal_percentage) if num_types > 1 else 0
        balance_score = 100 - (sum_diff / max_diff * 100) if max_diff > 0 else 100

        return {
            "type_distribution": relationship_type_distribution,
            "type_percentages": type_percentages,
            "balance_score": balance_score
        }

    def _calculate_graph_connectivity(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate graph connectivity metrics.

        Args:
            analysis: Dictionary containing analysis data including adjacency list and num_entities

        Returns:
            Dictionary containing graph connectivity metrics including a connectivity score
        """
        # Get entity count (use either "entity_count" or "num_entities")
        num_entities = 0
        if "entity_count" in analysis:
            num_entities = analysis["entity_count"]
        elif "num_entities" in analysis:
            num_entities = analysis["num_entities"]

        if num_entities == 0:
            return {
                "avg_degree": 0,
                "max_degree": 0,
                "isolated_entity_count": 0,
                "isolated_entity_percentage": 0,
                "connected_component_count": 0,
                "largest_component_size": 0,
                "largest_component_percentage": 0,
                "connectivity_score": 0,  # Add connectivity score
                "entities_without_relationships": 0,
                "entities_without_relationships_percentage": 0
            }

        # Get adjacency list and use precalculated isolated entity count
        adjacency_list = analysis["adjacency_list"]

        # 使用 _generate_analysis 计算的孤立实体数量
        isolated_entity_count = analysis.get("isolated_entity_count", 0)

        # 如果没有预先计算的值，则计算
        if isolated_entity_count == 0:
            # Calculate degree metrics
            degrees = {node: len(edges) for node, edges in adjacency_list.items()}
            # Count isolated entities (degree = 0)
            isolated_entity_count = sum(1 for degree in degrees.values() if degree == 0)
            # Add entities not in adjacency list
            isolated_entity_count += num_entities - len(degrees)

        isolated_entity_percentage = (isolated_entity_count / num_entities) * 100 if num_entities > 0 else 0

        # Calculate degree metrics
        degrees = {node: len(edges) for node, edges in adjacency_list.items()}
        total_degree = sum(degrees.values())
        avg_degree = total_degree / num_entities if num_entities > 0 else 0
        max_degree = max(degrees.values()) if degrees else 0

        # Find connected components
        connected_components = []
        visited = set()

        # 只遍历出现在邻接表中的节点
        nodes_to_check = set(adjacency_list.keys())

        for node in nodes_to_check:
            if node not in visited:
                # Start a new component
                component = set()
                queue = [node]
                visited.add(node)

                while queue:
                    current = queue.pop(0)
                    component.add(current)

                    # Add neighbors to queue
                    for neighbor in adjacency_list.get(current, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                connected_components.append(component)

        # Calculate connected component metrics
        connected_component_count = len(connected_components)

        largest_component_size = max(len(component) for component in connected_components) if connected_components else 0
        largest_component_percentage = (largest_component_size / num_entities) * 100 if num_entities > 0 else 0

        # Calculate a connectivity score (0-100)
        # Higher score means better connectivity
        # Factors: lower isolated percentage, fewer components, larger largest component
        isolation_factor = 100 - isolated_entity_percentage
        component_factor = 100 * (1 / connected_component_count) if connected_component_count > 0 else 0
        largest_component_factor = largest_component_percentage

        connectivity_score = (isolation_factor * 0.4 + component_factor * 0.2 + largest_component_factor * 0.4)

        return {
            "avg_degree": round(avg_degree, 2),
            "max_degree": max_degree,
            "isolated_entity_count": isolated_entity_count,
            "isolated_entity_percentage": round(isolated_entity_percentage, 2),
            "connected_component_count": connected_component_count,
            "largest_component_size": largest_component_size,
            "largest_component_percentage": round(largest_component_percentage, 2),
            "connectivity_score": round(connectivity_score, 2),  # Add connectivity score
            "entities_without_relationships": isolated_entity_count,
            "entities_without_relationships_percentage": round(isolated_entity_percentage, 2)
        }

    def _calculate_chapter_coverage(self, chapter_distribution: Dict[str, int]) -> Dict[str, Any]:
        """
        Calculate chapter coverage metrics.

        Args:
            chapter_distribution: Distribution of entities by chapter

        Returns:
            Dictionary containing chapter coverage metrics
        """
        total_entities = sum(chapter_distribution.values())

        # Calculate percentages for each chapter
        chapter_percentages = {
            chapter: count / total_entities * 100 if total_entities > 0 else 0
            for chapter, count in chapter_distribution.items()
        }

        # Calculate coverage score
        # A good coverage has entities from all chapters
        # We calculate how far the actual distribution is from the ideal
        num_chapters = len(chapter_distribution)
        ideal_percentage = 100 / num_chapters if num_chapters > 0 else 0

        # Calculate the sum of absolute differences from the ideal percentage
        sum_diff = sum(abs(percentage - ideal_percentage) for percentage in chapter_percentages.values())

        # Normalize to a 0-100 score (0 = completely unbalanced, 100 = perfectly balanced)
        # The maximum possible sum of differences is 2 * (100 - ideal_percentage)
        max_diff = 2 * (100 - ideal_percentage) if num_chapters > 1 else 0
        coverage_score = 100 - (sum_diff / max_diff * 100) if max_diff > 0 else 100

        return {
            "chapter_distribution": chapter_distribution,
            "chapter_percentages": chapter_percentages,
            "coverage_score": coverage_score
        }

    def _calculate_llm_quality_metrics(self, entities: List[Dict[str, Any]],
                                     relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        使用LLM计算知识图谱的质量指标: 置信度、清晰度和相关性

        Args:
            entities: 实体列表
            relationships: 关系列表

        Returns:
            包含LLM质量指标的字典
        """
        logger.info("开始使用LLM计算质量指标...")

        # 创建实体ID到实体的映射
        entity_map = {entity['id']: entity for entity in entities}

        # 评估结果
        confidence_scores = []
        clarity_scores = []
        relevance_scores = []

        # Sigmoid函数，用于将分数归一化到0-1区间
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        for rel in relationships:
            rel_id = rel.get('id', '')
            source_id = rel.get('source', '')
            target_id = rel.get('target', '')
            rel_type = rel.get('type', 'related_to')

            # 如果找不到源实体或目标实体，跳过
            if source_id not in entity_map or target_id not in entity_map:
                continue

            source_entity = entity_map[source_id]
            target_entity = entity_map[target_id]

            source_name = source_entity.get('name', '')
            target_name = target_entity.get('name', '')
            source_def = source_entity.get('definition', '')
            target_def = target_entity.get('definition', '')
            rel_def = rel.get('definition', '')

            # 构建提示，要求LLM评估关系的置信度、清晰度和相关性
            # 采用三个独立的提示来评估置信度、清晰度和相关性

            # 1. 置信度评估
            confidence_prompt = f"""
            Evaluate the descriptive accuracy of this computer networks triple:
            "{source_name} -> {rel_type} -> {target_name}"
            Source entity: "{source_name}" (Type: {source_entity.get('type', 'unknown')})
            Source definition: "{source_def}"
            Target entity: "{target_name}" (Type: {target_entity.get('type', 'unknown')})
            Target definition: "{target_def}"
            Relationship description: "{rel_def}"

            I need you to assess how accurate and precise the DESCRIPTIONS are for this triple (not whether the relationship itself is valid).

            Consider these factors:
            1. Are the entity definitions technically accurate and well-described?
            2. Does the relationship description accurately explain how these entities interact?
            3. Are the entity types appropriate and correctly assigned?
            4. Is the relationship type appropriate for how these entities actually relate?
            5. Are the descriptions free of technical errors or misconceptions?
            6. For each definition, is it a well-established technical fact?
            7. For each definition, is there substantial evidence in computer networking literature supporting this claim?
            8. For each definition, would computer networking experts broadly agree with this statement?
            9. For each definition, does it align with current technical understanding?

            Rate your confidence from 0.0 (completely uncertain) to 1.0 (extremely confident).


            This is about the QUALITY OF DESCRIPTIONS, not whether this relationship exists in reality.

            Rate the descriptive accuracy from 0.0 (poor descriptions) to 1.0 (excellent descriptions).
            Return only a single float value between 0.0 and 1.0.

            Example ratings:
            - 0.95-0.99: Perfect technical descriptions with overwhelming evidence, expertly written with precise terminology
            - 0.80-0.94: Very good descriptions with substantial supporting evidence
            - 0.60-0.79: Adequate descriptions with some evidence base
            - 0.40-0.59: Mediocre descriptions with some inaccuracies, mixed evidence
            - 0.20-0.39: Poor descriptions with significant technical errors, limited evidence
            - 0.01-0.19: Very poor descriptions with fundamental misconceptions,very weak evidence


            Return only the numeric value as a float between 0.0 and 1.0.
            """

            # 2. 清晰度评估
            clarity_prompt = f"""
            Evaluate the clarity and specificity of this computer networks relationship:
            "{source_name} -> {rel_type} -> {target_name}"
            Source definition: "{source_def}"
            Target definition: "{target_def}"
            Relationship description: "{rel_def}"

            Assess how clear, specific, and unambiguous this statement is to computer networking experts.

            Consider these aspects:
            1. Are the entities precise and unambiguous? (e.g., "TCP/IP" is clearer than "some protocols")
            2. Is the relationship type specific and well-defined? (e.g., "encapsulates" is clearer than "uses")
            3. Would networking experts interpret this statement consistently?
            4. Are there any vague terms or imprecise language that reduce clarity?
            5. Does the statement avoid unnecessary hedging words (e.g., "may", "possibly", "might")?

            Rate clarity from 0.0 (very ambiguous) to 1.0 (perfectly clear).
            Return only a single float value between 0.0 and 1.0.

            Example ratings:
            - 0.95-0.99: Crystal clear, highly specific statements with no ambiguity
            - 0.80-0.94: Very clear statements with minor room for interpretation
            - 0.60-0.79: Mostly clear statements with some potential ambiguity
            - 0.40-0.59: Statements with moderate ambiguity or vagueness
            - 0.20-0.39: Significantly vague or ambiguous statements
            - 0.01-0.19: Extremely vague statements with minimal specificity

            Return only the numeric value as a float between 0.0 and 1.0.
            """

            # 3. 相关性评估
            relevance_prompt = f"""
            Evaluate how relevant this relationship is to the computer networks domain:
            "{source_name} -> {rel_type} -> {target_name}"
            Source definition: "{source_def}"
            Target definition: "{target_def}"
            Relationship description: "{rel_def}"

            Assess how important and appropriate this statement is for inclusion in a specialized computer networks knowledge graph.

            Consider these aspects:
            1. Is this directly relevant to computer networks research or practice?
            2. Does this provide valuable information to computer networks experts?
            3. Would this knowledge be useful for computer networks applications (research, network design, troubleshooting, etc.)?
            4. Is this specialized knowledge rather than general world knowledge?
            5. Does it align with the core interests of the computer networks field?

            Rate relevance from 0.0 (completely irrelevant) to 1.0 (highly relevant).
            Return only a single float value between 0.0 and 1.0.

            Example ratings:
            - 0.95-0.99: Core computer networks knowledge essential to the field
            - 0.80-0.94: Highly relevant information for computer networks specialists
            - 0.60-0.79: Moderately relevant information with clear applications
            - 0.40-0.59: Somewhat relevant but not central to the domain
            - 0.20-0.39: Marginally relevant information
            - 0.01-0.19: Largely irrelevant to the computer networks domain

            Return only the numeric value as a float between 0.0 and 1.0.
            """

            try:
                # 调用LLM获取评估结果 - 置信度
                confidence_response = self.llm_client.call(confidence_prompt, temperature=0.1, response_format="text")
                try:
                    confidence_score = self._extract_float_score(confidence_response)
                except:
                    confidence_score = 0.5

                # 调用LLM获取评估结果 - 清晰度
                clarity_response = self.llm_client.call(clarity_prompt, temperature=0.1, response_format="text")
                try:
                    clarity_score = self._extract_float_score(clarity_response)
                except:
                    clarity_score = 0.5

                # 调用LLM获取评估结果 - 相关性
                relevance_response = self.llm_client.call(relevance_prompt, temperature=0.1, response_format="text")
                try:
                    relevance_score = self._extract_float_score(relevance_response)
                except:
                    relevance_score = 0.5

                # 直接使用0-1区间的分数，不需要转换
                confidence_scores.append(confidence_score)
                clarity_scores.append(clarity_score)
                relevance_scores.append(relevance_score)

                logger.info(f"评估关系 {rel_id} ({source_name} -> {target_name}): 置信度={confidence_score:.2f}, 清晰度={clarity_score:.2f}, 相关性={relevance_score:.2f}")

            except Exception as e:
                logger.error(f"评估关系 {rel_id} 时发生错误: {str(e)}")

        # 计算平均分数
        avg_confidence = sum(confidence_scores) / len(confidence_scores) * 100 if confidence_scores else 0
        avg_clarity = sum(clarity_scores) / len(clarity_scores) * 100 if clarity_scores else 0
        avg_relevance = sum(relevance_scores) / len(relevance_scores) * 100 if relevance_scores else 0

        # 计算整合分数 - 使用加权计算方式
        integration_threshold = 0.6  # 阈值Θ设为0.6

        # 新的加权计算方式: confidence*0.4 + clarity*0.2 + relevance*0.4
        integration_score = avg_confidence * 0.4 + avg_clarity * 0.2 + avg_relevance * 0.4

        # 在计算达标率时也使用相同的加权方式
        integration_rate = len([1 for c, l, r in zip(confidence_scores, clarity_scores, relevance_scores)
                            if (c * 0.4 + l * 0.2 + r * 0.4) >= integration_threshold]) / len(confidence_scores) * 100 if confidence_scores else 0

        logger.info(f"LLM质量评估结果: 平均置信度={avg_confidence:.2f}, 平均清晰度={avg_clarity:.2f}, 平均相关性={avg_relevance:.2f}")
        logger.info(f"集成评分: {integration_score:.2f} (加权计算: 置信度*0.4 + 清晰度*0.2 + 相关性*0.4)")
        logger.info(f"超过阈值的关系占比: {integration_rate:.2f}%")

        return {
            "average_confidence": {
                "score": round(avg_confidence, 2)
            },
            "average_clarity": {
                "score": round(avg_clarity, 2)
            },
            "average_relevance": {
                "score": round(avg_relevance, 2)
            },
            "integration_score": round(integration_score, 2),
            "integration_rate": round(integration_rate, 2),
            "sample_size": len(confidence_scores)
        }

    def _extract_float_score(self, content: str) -> float:
        """
        从LLM响应中提取浮点数值。

        Args:
            content: 包含浮点数值的文本

        Returns:
            解析并标准化到[0.0, 1.0]区间的浮点数值
        """
        # 查找文本中的浮点数
        import re
        float_matches = re.findall(r"[-+]?\d*\.\d+|\d+", content)
        if float_matches:
            try:
                score = float(float_matches[0])
                # 限制在[0.0, 1.0]区间内
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

        # 默认返回值
        return 0.5

    def _calculate_overall_score(self, entity_coverage: Dict[str, Any],
                               relationship_coverage: Dict[str, Any],
                               entity_type_balance: Dict[str, Any],
                               relationship_type_balance: Dict[str, Any],
                               graph_connectivity: Dict[str, Any],
                               chapter_coverage: Dict[str, Any],
                               manual_comparison: Dict[str, Any],
                               llm_metrics: Dict[str, Any] = None) -> float:
        """
        Calculate overall quality score.

        Args:
            entity_coverage: Entity coverage metrics
            relationship_coverage: Relationship coverage metrics
            entity_type_balance: Entity type balance metrics
            relationship_type_balance: Relationship type balance metrics
            graph_connectivity: Graph connectivity metrics
            chapter_coverage: Chapter coverage metrics
            manual_comparison: Manual comparison metrics
            llm_metrics: LLM-based quality metrics (optional)

        Returns:
            Overall quality score (0-100)
        """
        # Extract individual scores
        entity_coverage_score = entity_coverage["score"]
        relationship_coverage_score = relationship_coverage["score"]
        entity_type_balance_score = entity_type_balance["balance_score"]
        relationship_type_balance_score = relationship_type_balance["balance_score"]
        graph_connectivity_score = graph_connectivity["connectivity_score"]
        chapter_coverage_score = chapter_coverage["coverage_score"]
        manual_entity_coverage = manual_comparison["entity_coverage"]
        manual_relationship_coverage = manual_comparison["relationship_coverage"]

        # 将LLM指标纳入总分计算
        if llm_metrics and all(key in llm_metrics for key in ["average_confidence", "average_clarity", "average_relevance"]):
            confidence_score = llm_metrics["average_confidence"]["score"]
            clarity_score = llm_metrics["average_clarity"]["score"]
            relevance_score = llm_metrics["average_relevance"]["score"]

            # 更新权重，包含LLM指标
            overall_score = (
                0.12 * entity_coverage_score +
                0.12 * relationship_coverage_score +
                0.08 * entity_type_balance_score +
                0.08 * relationship_type_balance_score +
                0.08 * graph_connectivity_score +
                0.08 * chapter_coverage_score +
                0.12 * manual_entity_coverage +
                0.12 * manual_relationship_coverage +
                0.07 * confidence_score +
                0.07 * clarity_score +
                0.06 * relevance_score
            )
        else:
            # 原始权重，不包含LLM指标
            overall_score = (
                0.15 * entity_coverage_score +
                0.15 * relationship_coverage_score +
                0.1 * entity_type_balance_score +
                0.1 * relationship_type_balance_score +
                0.1 * graph_connectivity_score +
                0.1 * chapter_coverage_score +
                0.15 * manual_entity_coverage +
                0.15 * manual_relationship_coverage
            )

        return overall_score
