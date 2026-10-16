"""
Course knowledge graph database manager.
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Set, Tuple
from pathlib import Path
import networkx as nx

logger = logging.getLogger(__name__)

class CourseDBManager:
    """
    Class for managing the course knowledge graph database.
    """

    def __init__(self, db_dir: str):
        """
        Initialize the course DB manager.

        Args:
            db_dir: Directory for the course database
        """
        self.db_dir = db_dir
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        self.entity_file = os.path.join(db_dir, "entities.json")
        self.relationship_file = os.path.join(db_dir, "relationships.json")
        self.entity_types_file = os.path.join(db_dir, "entity_types.json")
        self.relationship_types_file = os.path.join(db_dir, "relationship_types.json")
        self.processed_files_list = os.path.join(db_dir, "processed_files.txt")
        logger.info(f"Initialized course DB manager with directory: {db_dir}")

    def load_entities(self) -> List[Dict[str, Any]]:
        """Load entities from the database."""
        if os.path.exists(self.entity_file):
            with open(self.entity_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def load_relationships(self) -> List[Dict[str, Any]]:
        """Load relationships from the database."""
        if os.path.exists(self.relationship_file):
            with open(self.relationship_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def load_entity_types(self) -> List[Dict[str, Any]]:
        """Load entity types from the database."""
        if os.path.exists(self.entity_types_file):
            with open(self.entity_types_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def load_relationship_types(self) -> List[Dict[str, Any]]:
        """Load relationship types from the database."""
        if os.path.exists(self.relationship_types_file):
            with open(self.relationship_types_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def save_entities(self, entities: List[Dict[str, Any]]) -> None:
        """Save entities to the database."""
        # We'll no longer directly save entities.json, see create_from_combined_graph method
        logger.info(f"Not directly saving entities to avoid overwriting files from the combined graph")

    def save_relationships(self, relationships: List[Dict[str, Any]]) -> None:
        """Save relationships to the database."""
        # We'll no longer directly save relationships.json, see create_from_combined_graph method
        logger.info(f"Not directly saving relationships to avoid overwriting files from the combined graph")

    def save_entity_types(self, entity_types: List[Dict[str, Any]]) -> None:
        """Save entity types to the database."""
        with open(self.entity_types_file, 'w', encoding='utf-8') as f:
            json.dump(entity_types, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(entity_types)} entity types to the course database")

    def save_relationship_types(self, relationship_types: List[Dict[str, Any]]) -> None:
        """Save relationship types to the database."""
        with open(self.relationship_types_file, 'w', encoding='utf-8') as f:
            json.dump(relationship_types, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(relationship_types)} relationship types to the course database")

    def is_file_processed(self, filename: str) -> bool:
        """Check if a file has been processed."""
        if not os.path.exists(self.processed_files_list):
            return False

        with open(self.processed_files_list, 'r', encoding='utf-8') as f:
            processed_files = f.read().splitlines()

        return filename in processed_files

    def mark_file_as_processed(self, filename: str) -> None:
        """Mark a file as processed."""
        if self.is_file_processed(filename):
            return

        with open(self.processed_files_list, 'a', encoding='utf-8') as f:
            f.write(f"{filename}\n")

        logger.info(f"Marked file {filename} as processed")

    def get_processed_files(self) -> List[str]:
        """Get list of processed files."""
        if not os.path.exists(self.processed_files_list):
            return []

        with open(self.processed_files_list, 'r', encoding='utf-8') as f:
            return f.read().splitlines()

    def merge_entities(self, new_entities: List[Dict[str, Any]], source_files: List[str]) -> List[Dict[str, Any]]:
        """
        Merge new entities with existing ones, handling duplicates.

        Args:
            new_entities: List of new entities
            source_files: List of source filenames

        Returns:
            List of merged entities
        """
        logger.info(f"Merging {len(new_entities)} new entities from {len(source_files)} files")

        existing_entities = self.load_entities()
        existing_entity_map = {entity["name"].lower(): entity for entity in existing_entities}

        merged_entities = existing_entities.copy()
        merge_stats = {
            "new_entities": 0,
            "updated_entities": 0,
            "conflicts_resolved": 0
        }

        for new_entity in new_entities:
            entity_name = new_entity["name"].lower()

            # 确保新实体的 occurrences 是新格式
            if "occurrences" not in new_entity or not isinstance(new_entity["occurrences"], list):
                source_file = source_files[0] if source_files else "unknown"
                page_number = new_entity.get("page_number", 0)
                new_entity["occurrences"] = [[f"{source_file} page{page_number}", "mention", 2]]

            if entity_name in existing_entity_map:
                # 实体已存在，处理合并冲突
                existing_entity = existing_entity_map[entity_name]
                logger.info(f"Resolving merge conflict for entity: {new_entity['name']}")

                # 初始化 occurrences 如果不存在或转换旧格式
                if "occurrences" not in existing_entity:
                    existing_entity["occurrences"] = []
                elif existing_entity["occurrences"] and isinstance(existing_entity["occurrences"][0], str):
                    old_occurrences = existing_entity["occurrences"]
                    existing_entity["occurrences"] = [[occ, "mention", 2] for occ in old_occurrences]

                # 合并新实体的 occurrences
                for occurrence in new_entity["occurrences"]:
                    # 检查是否已存在相同位置的记录
                    exists = False
                    for existing_occ in existing_entity["occurrences"]:
                        if existing_occ[0] == occurrence[0]:
                            # 相同位置，考虑函数类型的优先级
                            current_function = existing_occ[1]
                            new_function = occurrence[1]

                            # 函数类型优先级: introduction > mention > title
                            function_priority = {
                                "introduction": 3,
                                "mention": 2,
                                "title": 1
                            }

                            # 检查新功能类型是否有更高的优先级
                            if function_priority.get(new_function, 0) > function_priority.get(current_function, 0):
                                existing_occ[1] = new_function
                                existing_occ[2] = occurrence[2]
                                logger.info(f"Updated occurrence function for '{new_entity['name']}' at {occurrence[0]}: {current_function} -> {new_function}")

                            exists = True
                            break

                    if not exists:
                        existing_entity["occurrences"].append(occurrence)
                        logger.info(f"Added new occurrence for entity '{new_entity['name']}': {occurrence}")

                # 检查是否需要修正roadmap/TOC中的实体函数类型
                self._fix_roadmap_occurrences(existing_entity)

                # 更新定义（如果新实体有定义且现有实体没有）
                if "definition" in new_entity and new_entity["definition"] and (
                    "definition" not in existing_entity or not existing_entity["definition"]
                ):
                    existing_entity["definition"] = new_entity["definition"]
                    logger.info(f"Updated definition for entity '{new_entity['name']}'")

                # 基于最高权重的 occurrence 更新章节信息
                if existing_entity["occurrences"]:
                    sorted_occurrences = sorted(existing_entity["occurrences"], key=lambda x: (-x[2], x[0]))
                    highest_weighted_occurrence = sorted_occurrences[0]

                    # 从最高权重的出现位置提取章节信息
                    occurrence_parts = highest_weighted_occurrence[0].split()
                    if len(occurrence_parts) >= 1:
                        filename = occurrence_parts[0]
                        chapter_match = re.search(r'chap(\d+)', filename, re.IGNORECASE)
                        if chapter_match:
                            chapter_num = chapter_match.group(1)
                            existing_entity["chapter"] = f"Chapter {chapter_num}"
                            logger.info(f"Updated chapter for {existing_entity['name']} to {existing_entity['chapter']} based on weighted occurrence")

                merge_stats["updated_entities"] += 1
                merge_stats["conflicts_resolved"] += 1
            else:
                # 新实体，添加它
                clean_entity = {
                    "id": new_entity["id"],
                    "name": new_entity["name"],
                    "type": new_entity["type"],
                    "verified": new_entity.get("verified", False),
                    "definition": new_entity.get("definition", ""),
                    "occurrences": new_entity["occurrences"].copy()
                }

                # 检查是否需要修正roadmap/TOC中的实体函数类型
                self._fix_roadmap_occurrences(clean_entity)

                # 基于最高权重的 occurrence 设置章节信息
                if clean_entity["occurrences"]:
                    sorted_occurrences = sorted(clean_entity["occurrences"], key=lambda x: (-x[2], x[0]))
                    highest_weighted_occurrence = sorted_occurrences[0]

                    # 设置首次出现位置为最高权重的位置
                    clean_entity["first_occurrence_place"] = highest_weighted_occurrence[0]

                    # 从最高权重的出现位置提取章节信息
                    occurrence_parts = highest_weighted_occurrence[0].split()
                    if len(occurrence_parts) >= 1:
                        filename = occurrence_parts[0]
                        chapter_match = re.search(r'chap(\d+)', filename, re.IGNORECASE)
                        if chapter_match:
                            chapter_num = chapter_match.group(1)
                            clean_entity["chapter"] = f"Chapter {chapter_num}"
                        else:
                            clean_entity["chapter"] = new_entity.get("chapter", "")
                else:
                    clean_entity["chapter"] = new_entity.get("chapter", "")

                merged_entities.append(clean_entity)
                existing_entity_map[entity_name] = clean_entity
                logger.info(f"Added new entity '{new_entity['name']}' with occurrences {clean_entity['occurrences']}")
                merge_stats["new_entities"] += 1

        # 确保所有实体都有正确的字段结构
        for entity in merged_entities:
            # 移除不需要的字段
            for field_to_remove in ["all_occurrences", "first_occurrence_page", "first_occurrence", "page_number", "source_file"]:
                if field_to_remove in entity:
                    del entity[field_to_remove]

            # 如果没有 first_occurrence_place 但有 occurrences，使用最高权重的 occurrence
            if "first_occurrence_place" not in entity and "occurrences" in entity and entity["occurrences"]:
                sorted_occurrences = sorted(entity["occurrences"], key=lambda x: (-x[2], x[0]))
                entity["first_occurrence_place"] = sorted_occurrences[0][0]

        # 重新生成所有实体和关系的ID，确保唯一性
        merged_entities = self._regenerate_entity_ids(merged_entities)

        # 记录合并统计信息
        logger.info(f"Merge statistics: {merge_stats}")
        logger.info(f"Total entities after merge: {len(merged_entities)}")

        return merged_entities

    def merge_relationships(self, new_relationships: List[Dict[str, Any]],
                           entity_map: Dict[str, Dict[str, Any]], source_files: List[str]) -> List[Dict[str, Any]]:
        """
        Merge new relationships with existing ones, handling duplicates.
        支持新的entity1/entity2格式和传统的source/target格式

        Args:
            new_relationships: List of new relationships
            entity_map: Mapping of entity IDs to entities
            source_files: List of source filenames

        Returns:
            List of merged relationships
        """
        logger.info(f"Merging {len(new_relationships)} new relationships from {len(source_files)} files")

        existing_relationships = self.load_relationships()

        # 检查现有关系是否使用新格式(entity1/entity2)
        existing_format_is_new = False
        if existing_relationships and "entity1" in existing_relationships[0] and "entity2" in existing_relationships[0]:
            existing_format_is_new = True
            logger.info("检测到现有关系使用新的entity1/entity2格式")
        else:
            logger.info("检测到现有关系使用传统的source/target格式")

        # Create a dict of existing relationship keys to relationships
        existing_rel_map = {}
        for rel in existing_relationships:
            # 获取实体名称 - 适应新旧两种格式
            if existing_format_is_new:
                entity1_name = entity_map.get(rel["entity1"], {}).get("name", "").lower()
                entity2_name = entity_map.get(rel["entity2"], {}).get("name", "").lower()
            else:
                entity1_name = entity_map.get(rel["source"], {}).get("name", "").lower()
                entity2_name = entity_map.get(rel["target"], {}).get("name", "").lower()

            rel_type = rel["type"].lower()

            # 创建无向的实体对作为键
            entity_names = sorted([entity1_name, entity2_name])
            key = f"{entity_names[0]}|{rel_type}|{entity_names[1]}"
            existing_rel_map[key] = rel

        merged_relationships = existing_relationships.copy()
        merge_stats = {
            "new_relationships": 0,
            "updated_relationships": 0,
            "strength_updates": 0
        }

        for new_rel in new_relationships:
            # 检查新关系是否使用新格式(entity1/entity2)
            using_new_format = "entity1" in new_rel and "entity2" in new_rel

            # 获取实体名称 - 适应新旧两种格式
            if using_new_format:
                entity1_name = entity_map.get(new_rel["entity1"], {}).get("name", "").lower()
                entity2_name = entity_map.get(new_rel["entity2"], {}).get("name", "").lower()
            else:
                entity1_name = entity_map.get(new_rel["source"], {}).get("name", "").lower()
                entity2_name = entity_map.get(new_rel["target"], {}).get("name", "").lower()

            rel_type = new_rel["type"].lower()

            if not entity1_name or not entity2_name:
                logger.warning(f"Skipping relationship with missing entity: {new_rel}")
                continue

            # 创建无向的实体对作为键
            entity_names = sorted([entity1_name, entity2_name])
            key = f"{entity_names[0]}|{rel_type}|{entity_names[1]}"

            if key in existing_rel_map:
                # Relationship already exists, update based on evidence
                existing_rel = existing_rel_map[key]
                new_strength = new_rel.get("strength", 1)
                existing_strength = existing_rel.get("strength", 1)

                # 计算新的强度值，考虑多个因素
                strength_factors = {
                    "occurrence_count": len(new_rel.get("occurrences", [])),
                    "page_proximity": self._calculate_page_proximity(new_rel, existing_rel),
                    "chapter_match": 1 if new_rel.get("chapter") == existing_rel.get("chapter") else 0.5
                }

                # 计算加权强度
                weighted_strength = (
                    new_strength * 0.4 +  # 新关系的原始强度
                    existing_strength * 0.3 +  # 现有关系的强度
                    strength_factors["occurrence_count"] * 0.1 +  # 出现次数
                    strength_factors["page_proximity"] * 0.1 +  # 页面接近度
                    strength_factors["chapter_match"] * 0.1  # 章节匹配
                )

                # 更新强度
                if weighted_strength > existing_strength:
                    existing_rel["strength"] = min(3, round(weighted_strength, 1))
                    logger.info(f"Updated relationship strength for {entity1_name} and {entity2_name} from {existing_strength} to {existing_rel['strength']}")
                    merge_stats["strength_updates"] += 1

                # 如果新关系有定义，而现有关系没有，添加定义
                if "definition" in new_rel and new_rel.get("definition") and (
                    "definition" not in existing_rel or not existing_rel.get("definition")
                ):
                    existing_rel["definition"] = new_rel["definition"]
                    logger.info(f"Updated definition for relationship between {entity1_name} and {entity2_name}")

                merge_stats["updated_relationships"] += 1
            else:
                # New relationship, add it with cleaned structure adapting to proper format
                clean_rel = {}

                # 保留共同基础属性
                clean_rel["id"] = new_rel["id"]
                clean_rel["type"] = new_rel["type"]
                clean_rel["strength"] = new_rel.get("strength", 1)
                clean_rel["page_number"] = new_rel.get("page_number", 0)
                clean_rel["chapter"] = new_rel.get("chapter", "")
                clean_rel["definition"] = new_rel.get("definition", "")

                # 根据当前系统使用的格式添加特定字段
                if existing_format_is_new or (len(merged_relationships) == 0):
                    # 使用新格式 entity1/entity2
                    if using_new_format:
                        # 新关系已经是新格式
                        clean_rel["entity1"] = new_rel["entity1"]
                        clean_rel["entity2"] = new_rel["entity2"]
                        clean_rel["entity1_name"] = entity_map.get(new_rel["entity1"], {}).get("name", "")
                        clean_rel["entity2_name"] = entity_map.get(new_rel["entity2"], {}).get("name", "")
                    else:
                        # 将旧格式转换为新格式
                        clean_rel["entity1"] = new_rel["source"]
                        clean_rel["entity2"] = new_rel["target"]
                        clean_rel["entity1_name"] = entity_map.get(new_rel["source"], {}).get("name", "")
                        clean_rel["entity2_name"] = entity_map.get(new_rel["target"], {}).get("name", "")
                else:
                    # 使用旧格式 source/target
                    if using_new_format:
                        # 将新格式转换为旧格式
                        clean_rel["source"] = new_rel["entity1"]
                        clean_rel["target"] = new_rel["entity2"]
                        clean_rel["source_name"] = entity_map.get(new_rel["entity1"], {}).get("name", "")
                        clean_rel["target_name"] = entity_map.get(new_rel["entity2"], {}).get("name", "")
                    else:
                        # 保持旧格式
                        clean_rel["source"] = new_rel["source"]
                        clean_rel["target"] = new_rel["target"]
                        clean_rel["source_name"] = entity_map.get(new_rel["source"], {}).get("name", "")
                        clean_rel["target_name"] = entity_map.get(new_rel["target"], {}).get("name", "")

                merged_relationships.append(clean_rel)
                existing_rel_map[key] = clean_rel

                if existing_format_is_new:
                    logger.info(f"Added new relationship: {clean_rel.get('entity1_name', '')} and {clean_rel.get('entity2_name', '')} with strength {clean_rel['strength']}")
                else:
                    logger.info(f"Added new relationship: {clean_rel.get('source_name', '')} -> {clean_rel.get('target_name', '')} with strength {clean_rel['strength']}")

                merge_stats["new_relationships"] += 1

        # 确保所有关系都有正确的字段结构
        for rel in merged_relationships:
            # 移除不需要的字段
            if "verified" in rel:
                del rel["verified"]
            if "source_file" in rel:
                del rel["source_file"]

            # 确保所有关系都有强度属性
            if "strength" not in rel:
                rel["strength"] = 1

                if existing_format_is_new:
                    logger.info(f"Added default strength 1 to relationship: {rel.get('entity1_name', '')} and {rel.get('entity2_name', '')}")
                else:
                    logger.info(f"Added default strength 1 to relationship: {rel.get('source_name', '')} -> {rel.get('target_name', '')}")
            else:
                # 检查如果definition字段中包含strength信息但strength字段是1，尝试提取并更新
                if rel.get("strength") == 1 and "definition" in rel and rel["definition"]:
                    strength_match = re.search(r'[Ss]trength:\s*(\d+)', rel["definition"])
                    if strength_match:
                        try:
                            strength_value = int(strength_match.group(1))
                            if 1 <= strength_value <= 3 and strength_value > rel["strength"]:
                                rel["strength"] = strength_value

                                if existing_format_is_new:
                                    logger.info(f"Updated strength from definition for {rel.get('entity1_name', '')} and {rel.get('entity2_name', '')} to {strength_value}")
                                else:
                                    logger.info(f"Updated strength from definition for {rel.get('source_name', '')} -> {rel.get('target_name', '')} to {strength_value}")
                        except ValueError:
                            pass

            # 确保所有关系都有实体名称
            if existing_format_is_new:
                # 新格式 - 确保entity1_name和entity2_name存在
                if "entity1" in rel and "entity1_name" not in rel and rel["entity1"] in entity_map:
                    rel["entity1_name"] = entity_map[rel["entity1"]]["name"]
                if "entity2" in rel and "entity2_name" not in rel and rel["entity2"] in entity_map:
                    rel["entity2_name"] = entity_map[rel["entity2"]]["name"]
            else:
                # 旧格式 - 确保source_name和target_name存在
                if "source" in rel and "source_name" not in rel and rel["source"] in entity_map:
                    rel["source_name"] = entity_map[rel["source"]]["name"]
                if "target" in rel and "target_name" not in rel and rel["target"] in entity_map:
                    rel["target_name"] = entity_map[rel["target"]]["name"]

        # 重新生成所有关系的ID，确保唯一性
        merged_relationships = self._regenerate_relationship_ids(merged_relationships)

        # 记录合并统计信息
        logger.info(f"Merge statistics: {merge_stats}")
        logger.info(f"Total relationships after merge: {len(merged_relationships)}")

        return merged_relationships

    def _calculate_page_proximity(self, rel1: Dict[str, Any], rel2: Dict[str, Any]) -> float:
        """
        Calculate the proximity between two relationships based on their page numbers.

        Args:
            rel1: First relationship
            rel2: Second relationship

        Returns:
            Proximity score between 0 and 1
        """
        page1 = rel1.get("page_number", 0)
        page2 = rel2.get("page_number", 0)

        if page1 == page2:
            return 1.0
        elif abs(page1 - page2) <= 2:
            return 0.8
        elif abs(page1 - page2) <= 5:
            return 0.5
        else:
            return 0.2

    def _regenerate_entity_ids(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        为所有实体重新生成ID，确保唯一性。

        Args:
            entities: 实体列表

        Returns:
            更新后的实体列表
        """
        id_map = {}  # 旧ID -> 新ID的映射

        # 生成新ID
        for i, entity in enumerate(entities):
            old_id = entity["id"]
            new_id = f"e{i+1:04d}"  # 格式：e0001, e0002, ...
            id_map[old_id] = new_id
            entity["id"] = new_id

        logger.info(f"Regenerated IDs for {len(entities)} entities")
        return entities

    def _regenerate_relationship_ids(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        为所有关系重新生成ID，确保唯一性。

        Args:
            relationships: 关系列表

        Returns:
            更新后的关系列表
        """
        # 生成新ID
        for i, relationship in enumerate(relationships):
            relationship["id"] = f"r{i+1:04d}"  # 格式：r0001, r0002, ...

        logger.info(f"Regenerated IDs for {len(relationships)} relationships")
        return relationships

    def _regenerate_all_ids(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        为所有实体和关系重新生成ID，并更新关系中的引用。

        Args:
            entities: 实体列表
            relationships: 关系列表

        Returns:
            更新后的实体和关系元组
        """
        # 首先为实体重新生成ID并创建映射
        id_map = {}  # 旧ID -> 新ID的映射

        for i, entity in enumerate(entities):
            old_id = entity["id"]
            new_id = f"e{i+1:04d}"  # 格式：e0001, e0002, ...
            id_map[old_id] = new_id
            entity["id"] = new_id

        # 然后更新关系中的实体引用并生成新的关系ID
        for i, relationship in enumerate(relationships):
            # 更新source和target
            if relationship["source"] in id_map:
                relationship["source"] = id_map[relationship["source"]]
            else:
                logger.warning(f"Source entity ID {relationship['source']} not found in entity ID map")

            if relationship["target"] in id_map:
                relationship["target"] = id_map[relationship["target"]]
            else:
                logger.warning(f"Target entity ID {relationship['target']} not found in entity ID map")

            # 生成新的关系ID
            relationship["id"] = f"r{i+1:04d}"  # 格式：r0001, r0002, ...

        logger.info(f"Regenerated all IDs for {len(entities)} entities and {len(relationships)} relationships")
        return entities, relationships

    def merge_entity_types(self, new_types: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge new entity types with existing ones."""
        existing_types = self.load_entity_types()
        existing_type_names = {t["name"].lower() for t in existing_types}

        merged_types = existing_types.copy()

        for new_type in new_types:
            if new_type["name"].lower() not in existing_type_names:
                merged_types.append(new_type)
                existing_type_names.add(new_type["name"].lower())
                logger.info(f"Added new entity type: {new_type['name']}")

        return merged_types

    def merge_relationship_types(self, new_types: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge new relationship types with existing ones."""
        existing_types = self.load_relationship_types()
        existing_type_names = {t["name"].lower() for t in existing_types}

        merged_types = existing_types.copy()

        for new_type in new_types:
            if new_type["name"].lower() not in existing_type_names:
                merged_types.append(new_type)
                existing_type_names.add(new_type["name"].lower())
                logger.info(f"Added new relationship type: {new_type['name']}")

        return merged_types

    def create_from_combined_graph(self, combined_graph_path: str) -> None:
        """
        从combined_graph.json文件创建实体和关系文件

        Args:
            combined_graph_path: combined_graph.json文件的路径
        """
        if not os.path.exists(combined_graph_path):
            logger.error(f"Combined graph file not found: {combined_graph_path}")
            return

        logger.info(f"Creating entities and relationships files from {combined_graph_path}")

        try:
            with open(combined_graph_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            entities = data.get("entities", [])
            relationships = data.get("relationships", [])

            # 确保实体和关系有正确的relationship_count
            # 重新计算一次关系计数，确保准确性
            entity_relationship_counts = self._calculate_relationship_counts(relationships)

            # 更新每个实体的relationship_count
            for entity in entities:
                entity_id = entity["id"]
                entity["relationship_count"] = entity_relationship_counts.get(entity_id, 0)

            # 记录实体的关系计数
            logger.info("Entity relationship counts after recalculation:")
            for entity in entities:
                logger.info(f"Entity {entity['name']} (ID: {entity['id']}) has {entity['relationship_count']} relationships")

            # 确保entities和relationships目录存在
            Path(self.db_dir).mkdir(parents=True, exist_ok=True)

            # 删除之前可能存在的文件
            if os.path.exists(self.entity_file):
                os.remove(self.entity_file)
                logger.info(f"Removed existing entities.json file")

            if os.path.exists(self.relationship_file):
                os.remove(self.relationship_file)
                logger.info(f"Removed existing relationships.json file")

            # 写入新文件
            with open(self.entity_file, 'w', encoding='utf-8') as f:
                json.dump(entities, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(entities)} entities from combined graph to {self.entity_file}")

            with open(self.relationship_file, 'w', encoding='utf-8') as f:
                json.dump(relationships, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(relationships)} relationships from combined graph to {self.relationship_file}")

            # 更新实体类型和关系类型
            self._update_entity_types(entities)
            self._update_relationship_types(relationships)

        except Exception as e:
            logger.error(f"Error creating files from combined graph: {str(e)}")

    def _calculate_relationship_counts(self, relationships: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        计算每个实体在关系中出现的次数

        Args:
            relationships: 关系列表

        Returns:
            实体ID到关系计数的映射
        """
        entity_counts = {}

        # 遍历所有关系，为每个实体计数
        for rel in relationships:
            source_id = rel.get("source")
            target_id = rel.get("target")

            if source_id:
                entity_counts[source_id] = entity_counts.get(source_id, 0) + 1

            if target_id:
                entity_counts[target_id] = entity_counts.get(target_id, 0) + 1

        return entity_counts

    def _update_entity_types(self, entities: List[Dict[str, Any]]) -> None:
        """
        从实体列表中更新实体类型文件

        Args:
            entities: 实体列表
        """
        # 首先从combined_new_types.json中加载新发现的实体类型定义
        new_types_path = os.path.join(os.path.dirname(self.db_dir), "exploration", "combined_new_types.json")
        new_types = {}
        new_type_colors = {}
        if os.path.exists(new_types_path):
            try:
                with open(new_types_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for type_info in data.get("new_entity_types", []):
                        new_types[type_info["name"]] = type_info["definition"]
                        if "color" in type_info:
                            new_type_colors[type_info["name"]] = type_info["color"]
            except Exception as e:
                logger.error(f"Error loading new entity types: {str(e)}")

        # 收集所有实体类型
        entity_types = set()
        for entity in entities:
            entity_type = entity.get("type", "")
            if entity_type:
                entity_types.add(entity_type)

        # 默认颜色映射，为没有指定颜色的实体类型提供默认值
        default_colors = {
            "concept": "#1f77b4",            # 蓝色
            "technology": "#ff7f0e",         # 橙色
            "physical_media": "#2ca02c",     # 绿色
            "NetworkArchitecture": "#d62728", # 红色
            "NetworkTopology": "#9467bd",    # 紫色
            "CommunicationMethod": "#8c564b", # 棕色
            "other": "#7f7f7f"               # 灰色
        }

        # 创建实体类型对象，包含定义和颜色
        entity_type_objects = []
        for et in sorted(entity_types):
            definition = new_types.get(et, f"Standard entity type: {et}")
            # 首先尝试从新类型中获取颜色，如果没有则使用默认颜色映射，最后使用灰色作为兜底颜色
            color = new_type_colors.get(et, default_colors.get(et, "#7f7f7f"))
            entity_type_objects.append({
                "name": et,
                "definition": definition,
                "color": color
            })

        with open(self.entity_types_file, 'w', encoding='utf-8') as f:
            json.dump(entity_type_objects, f, indent=2, ensure_ascii=False)
        logger.info(f"Updated {len(entity_type_objects)} entity types from combined graph")

    def _update_relationship_types(self, relationships: List[Dict[str, Any]]) -> None:
        """
        从关系列表中更新关系类型文件

        Args:
            relationships: 关系列表
        """
        # 首先从combined_new_types.json中加载新发现的关系类型定义
        new_types_path = os.path.join(os.path.dirname(self.db_dir), "exploration", "combined_new_types.json")
        new_types = {}
        new_type_colors = {}
        if os.path.exists(new_types_path):
            try:
                with open(new_types_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for type_info in data.get("new_relationship_types", []):
                        new_types[type_info["name"]] = type_info["definition"]
                        if "color" in type_info:
                            new_type_colors[type_info["name"]] = type_info["color"]
            except Exception as e:
                logger.error(f"Error loading new relationship types: {str(e)}")

        # 收集所有关系类型
        relationship_types = set()
        for rel in relationships:
            rel_type = rel.get("type", "")
            if rel_type:
                relationship_types.add(rel_type)

        # 默认颜色映射
        default_colors = {
            "related_to": "#1f77b4"  # 蓝色作为默认颜色
        }

        # 创建关系类型对象，包含定义和颜色
        rel_type_objects = []
        for rt in sorted(relationship_types):
            definition = new_types.get(rt, f"Standard relationship type: {rt}")
            # 首先尝试从新类型中获取颜色，如果没有则使用默认颜色
            color = new_type_colors.get(rt, default_colors.get(rt, "#1f77b4"))
            rel_type_objects.append({
                "name": rt,
                "definition": definition,
                "color": color
            })

        with open(self.relationship_types_file, 'w', encoding='utf-8') as f:
            json.dump(rel_type_objects, f, indent=2, ensure_ascii=False)
        logger.info(f"Updated {len(rel_type_objects)} relationship types from combined graph")

    def validate_merge(self, merged_entities: List[Dict[str, Any]],
                      merged_relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the merged knowledge graph to ensure its quality.

        Args:
            merged_entities: List of merged entities
            merged_relationships: List of merged relationships

        Returns:
            Dictionary containing validation results and statistics
        """
        logger.info("Validating merged knowledge graph...")

        validation_results = {
            "entity_validation": {
                "total_entities": len(merged_entities),
                "entities_with_definition": 0,
                "entities_with_chapter": 0,
                "entities_with_occurrences": 0,
                "entities_without_relationships": []
            },
            "relationship_validation": {
                "total_relationships": len(merged_relationships),
                "relationships_with_definition": 0,
                "relationships_with_strength": 0,
                "relationships_with_chapter": 0,
                "invalid_relationships": []
            },
            "graph_validation": {
                "connected_components": 0,
                "average_degree": 0,
                "isolated_entities": 0
            }
        }

        # 验证实体
        entity_map = {entity["id"]: entity for entity in merged_entities}
        entity_relationship_count = {entity["id"]: 0 for entity in merged_entities}

        for entity in merged_entities:
            if "definition" in entity and entity["definition"]:
                validation_results["entity_validation"]["entities_with_definition"] += 1

            if "chapter" in entity and entity["chapter"]:
                validation_results["entity_validation"]["entities_with_chapter"] += 1

            if "occurrences" in entity and entity["occurrences"]:
                validation_results["entity_validation"]["entities_with_occurrences"] += 1

        # 验证关系
        for rel in merged_relationships:
            # 检查源实体和目标实体是否存在
            if rel["source"] not in entity_map or rel["target"] not in entity_map:
                validation_results["relationship_validation"]["invalid_relationships"].append({
                    "id": rel["id"],
                    "reason": "Missing source or target entity"
                })
                continue

            # 更新实体关系计数
            entity_relationship_count[rel["source"]] += 1
            entity_relationship_count[rel["target"]] += 1

            if "definition" in rel and rel["definition"]:
                validation_results["relationship_validation"]["relationships_with_definition"] += 1

            if "strength" in rel and rel["strength"] > 0:
                validation_results["relationship_validation"]["relationships_with_strength"] += 1

            if "chapter" in rel and rel["chapter"]:
                validation_results["relationship_validation"]["relationships_with_chapter"] += 1

        # 检查孤立实体
        for entity_id, count in entity_relationship_count.items():
            if count == 0:
                validation_results["entity_validation"]["entities_without_relationships"].append(entity_id)

        # 计算图的基本统计信息
        G = nx.DiGraph()

        # 添加节点
        for entity in merged_entities:
            G.add_node(entity["id"])

        # 添加边
        for rel in merged_relationships:
            if rel["source"] in entity_map and rel["target"] in entity_map:
                G.add_edge(rel["source"], rel["target"])

        # 计算连通分量
        validation_results["graph_validation"]["connected_components"] = nx.number_weakly_connected_components(G)

        # 计算平均度数
        if G.number_of_nodes() > 0:
            validation_results["graph_validation"]["average_degree"] = sum(dict(G.degree()).values()) / G.number_of_nodes()

        # 计算孤立实体
        validation_results["graph_validation"]["isolated_entities"] = len(validation_results["entity_validation"]["entities_without_relationships"])

        # 记录验证结果
        logger.info("Validation results:")
        logger.info(f"Entities: {validation_results['entity_validation']}")
        logger.info(f"Relationships: {validation_results['relationship_validation']}")
        logger.info(f"Graph: {validation_results['graph_validation']}")

        return validation_results

    def identify_new_entities_and_relationships(self, file_entities: List[Dict[str, Any]],
                                                 file_relationships: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Identify new entities and relationships that are not in the existing knowledge graph.

        Args:
            file_entities: List of entities from the current file
            file_relationships: List of relationships from the current file

        Returns:
            Tuple of (new_entities, new_relationships)
        """
        logger.info("Identifying new entities and relationships...")

        # Load existing entities and relationships
        existing_entities = self.load_entities()
        existing_relationships = self.load_relationships()

        # Create maps for quick lookup
        existing_entity_map = {entity["name"].lower(): entity for entity in existing_entities}
        existing_relationship_map = {
            f"{rel['source']}_{rel['type']}_{rel['target']}": rel
            for rel in existing_relationships
        }

        # Identify new entities
        new_entities = []
        for entity in file_entities:
            if entity["name"].lower() not in existing_entity_map:
                new_entities.append(entity)
                logger.info(f"New entity found: {entity['name']}")

        # Identify new relationships
        new_relationships = []
        for rel in file_relationships:
            rel_key = f"{rel['source']}_{rel['type']}_{rel['target']}"
            if rel_key not in existing_relationship_map:
                new_relationships.append(rel)
                logger.info(f"New relationship found: {rel['source']} -> {rel['target']} ({rel['type']})")

        logger.info(f"Found {len(new_entities)} new entities and {len(new_relationships)} new relationships")
        return new_entities, new_relationships

    def _fix_roadmap_occurrences(self, entity: Dict[str, Any]) -> None:
        """
        修复roadmap/TOC中的实体函数类型，确保正确分类为title而不是mention。

        Args:
            entity: 要修复的实体
        """
        if "occurrences" not in entity or not entity["occurrences"]:
            return

        # 早期页面（通常是前几页）更可能是roadmap/TOC
        roadmap_pages = {}
        non_roadmap_pages = {}

        # 按照文件分组统计，并识别早期页面
        for occurrence in entity["occurrences"]:
            location = occurrence[0]  # 例如: "chap1-1_1-10 page2"
            function = occurrence[1]  # 例如: "mention", "title", "introduction"

            # 提取文件名和页码
            parts = location.split(" page")
            if len(parts) != 2:
                continue

            filename = parts[0]
            try:
                page_num = int(parts[1])
            except ValueError:
                continue

            # 页码小于等于3的早期页面更可能是roadmap
            is_early_page = page_num <= 3

            if is_early_page:
                roadmap_pages[location] = occurrence
            else:
                non_roadmap_pages[location] = occurrence

        # 检查是否有早期页面被标记为"mention"，但实际上应该是"title"
        # 特别是如果所有其他出现都是"mention"，那么早期页面很可能是roadmap中的"title"
        if roadmap_pages and all(occ[1] == "mention" for occ in entity["occurrences"]):
            # 修改早期页面的函数类型为"title"
            for location, occurrence in roadmap_pages.items():
                if occurrence[1] == "mention":
                    occurrence[1] = "title"
                    occurrence[2] = 1  # 更新权重
                    logger.info(f"Fixed occurrence function for '{entity['name']}' at {location}: mention -> title (likely roadmap/TOC)")

        # 最后再次按照权重排序出现位置
        entity["occurrences"] = sorted(entity["occurrences"], key=lambda x: (-x[2], x[0]))
