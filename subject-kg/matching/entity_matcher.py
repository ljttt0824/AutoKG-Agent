"""
Entity matcher for verifying entities against the original text.
"""

import re
import logging
import uuid
from typing import Dict, List, Any, Tuple, Set

from tqdm import tqdm

import config

logger = logging.getLogger(__name__)

class EntityMatcher:
    """
    Class for matching and verifying entities against the original text.
    """

    def __init__(self):
        """
        Initialize the entity matcher.
        """
        logger.info("Initialized entity matcher")

    def _filter_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        二次筛选实体，去掉名字、学校、网址、日期、IP地址、具体数值等。

        Args:
            entities: 要筛选的实体列表

        Returns:
            筛选后的实体列表
        """
        logger.info(f"开始二次筛选实体，原有实体数量: {len(entities)}")

        # 定义筛选规则
        filter_patterns = [
            # 数字+单位模式
            r'^\d+(\.\d+)?\s*(Mbps|Kbps|Gbps|bits?|bytes?|KB|MB|GB|TB)$',
            # 频率值
            r'^\d+(\.\d+)?\s*(MHz|GHz|kHz|Hz)$',
            # IEEE标准
            r'^IEEE\s+\d+(\.\d+[a-z]?)?$',
            # RFC标准号
            r'^RFC\s*\d+$',
            r'^RFCs?\s*\d+$',
            # IP地址和CIDR表示法
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(/\d{1,2})?$',
            r'^[0-9a-fA-F:]+(/\d{1,2})?$',  # IPv6地址
            # 子网掩码
            r'^mask\s+\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            # 数字+位/比特/字节等
            r'^\d+\s*bits?\s+of\s+.*$',
            # 802.11系列标准
            r'^802\.\d+[a-z]?$',
            # 纯数字
            r'^\d+(\.\d+)?$',
            # 时间戳或日期
            r'^\d{1,2}[:/-]\d{1,2}([:/-]\d{2,4})?$',
            r'^\d{4}[:/-]\d{1,2}[:/-]\d{1,2}$',
            # 波段和信道
            r'^band\s+\d+$',
            r'^channel\s+\d+$',
            # 版本号
            r'^v\d+(\.\d+)*$',
            r'^version\s+\d+(\.\d+)*$'
        ]

        # 编译正则表达式
        compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in filter_patterns]

        # 其他特定过滤词
        specific_filters = [
            '128 bit', '802.11g', 'IEEE 802.11', '192.168.25.0', '1.5 Mbps',
            'mask 255.255.255.0', '28.8 Kbps', 'RFC 2326', 'RFC 2459',
            'RFCs 2401', '48 bits of key', '211.80.32.0/19', '1920 MHz',
            '1935 MHz', '1940 MHz', '1955 MHz', '1980 MHz', '2010 MHz',
            '2025 MHz', '2110 MHz', '2125 MHz', '2130 MHz', '2145 MHz',
            '2170 MHz'
        ]

        # 检查实体类型为physical_media且名称包含数字+MHz/GHz的实体
        def is_frequency_entity(entity):
            if entity.get("type") == "physical_media":
                name = entity.get("name", "").lower()
                if any(unit in name for unit in ["mhz", "ghz", "khz", "hz"]) and any(c.isdigit() for c in name):
                    return True
            return False

        # 筛选实体
        filtered_entities = []
        skipped_entities = []
        reasons = []  # 记录过滤原因

        for entity in entities:
            entity_name = entity["name"].strip()

            # 检查是否匹配任何模式
            should_skip = False
            skip_reason = ""

            # 检查特定过滤词
            if entity_name.lower() in [f.lower() for f in specific_filters]:
                skipped_entities.append(entity_name)
                skip_reason = "特定过滤词匹配"
                should_skip = True
                reasons.append((entity_name, skip_reason))
                continue

            # 检查正则表达式模式
            for pattern in compiled_patterns:
                if pattern.match(entity_name):
                    skipped_entities.append(entity_name)
                    skip_reason = f"正则模式匹配: {pattern.pattern}"
                    should_skip = True
                    reasons.append((entity_name, skip_reason))
                    break

            # 检查是否为频率实体
            if not should_skip and is_frequency_entity(entity):
                skipped_entities.append(entity_name)
                skip_reason = "频率值实体"
                should_skip = True
                reasons.append((entity_name, skip_reason))
                continue

            if not should_skip:
                filtered_entities.append(entity)

        # 输出统计信息
        filtered_count = len(entities) - len(filtered_entities)
        logger.info(f"二次筛选共过滤掉 {filtered_count} 个实体，占比 {filtered_count/len(entities)*100:.2f}%")

        # 记录被过滤的实体前三个示例
        if skipped_entities:
            top_three = reasons[:3]
            detailed_examples = "\n".join([f"- {name}: {reason}" for name, reason in top_three])
            logger.info(f"前三个被过滤的实体及原因:\n{detailed_examples}")

            # 记录更多被过滤的实体
            if len(skipped_entities) > 3:
                logger.info(f"其他被过滤的实体示例: {', '.join(skipped_entities[3:10])}" +
                          (f" 等共 {len(skipped_entities)} 个" if len(skipped_entities) > 10 else ""))

        logger.info(f"二次筛选完成，筛选前: {len(entities)} 实体，筛选后: {len(filtered_entities)} 实体")
        return filtered_entities

    def match_entities(self, entities_to_match: List[Dict[str, Any]], existing_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Match new entities with existing entities and merge them when appropriate.

        Args:
            entities_to_match: List of new entities to match
            existing_entities: List of existing entities to match against

        Returns:
            List of merged entities
        """
        logger.info(f"Matching {len(entities_to_match)} new entities against {len(existing_entities)} existing entities")

        # If no existing entities, just return the new ones
        if not existing_entities:
            return entities_to_match

        merged_entities = existing_entities.copy()

        # Map of entity names to indices in the merged list for fast lookup
        entity_name_map = {}
        for idx, entity in enumerate(merged_entities):
            name = entity["name"].lower()
            if name not in entity_name_map:
                entity_name_map[name] = []
            entity_name_map[name].append(idx)

        # For each new entity, check if it already exists and merge if necessary
        for new_entity in entities_to_match:
            new_name = new_entity["name"].lower()

            # Check if an entity with this name already exists
            if new_name in entity_name_map:
                # Entity exists, update its occurrences
                for idx in entity_name_map[new_name]:
                    existing_entity = merged_entities[idx]

                    # Merge occurrences
                    if "occurrences" in new_entity and "occurrences" in existing_entity:
                        existing_occurrences = {tuple(occ) for occ in existing_entity["occurrences"]}
                        for occ in new_entity["occurrences"]:
                            existing_occurrences.add(tuple(occ))
                        existing_entity["occurrences"] = [list(occ) for occ in existing_occurrences]

                    # Update first_occurrence flag if needed
                    if new_entity.get("first_occurrence", False) and not existing_entity.get("first_occurrence", False):
                        existing_entity["first_occurrence"] = True

                    # Merge other fields as needed (e.g., definition, type)
                    if "definition" in new_entity and "definition" not in existing_entity:
                        existing_entity["definition"] = new_entity["definition"]

                    if "type" in new_entity and new_entity["type"] != "other" and (
                            "type" not in existing_entity or existing_entity["type"] == "other"):
                        existing_entity["type"] = new_entity["type"]
            else:
                # New entity, add it to the merged list
                merged_entities.append(new_entity)

                # Update the name map
                if new_name not in entity_name_map:
                    entity_name_map[new_name] = []
                entity_name_map[new_name].append(len(merged_entities) - 1)

        logger.info(f"Merged entities: {len(merged_entities)}")
        return merged_entities

    def match_relationships(self, relationships_to_match: List[Dict[str, Any]], merged_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Match relationships with existing entities and ensure entities are valid.
        Prevent duplicate relationships between the same entity pair, regardless of direction.

        Args:
            relationships_to_match: List of relationships to process
            merged_entities: List of merged entities to use for entity validation

        Returns:
            List of valid deduplicated relationships
        """
        logger.info(f"Processing {len(relationships_to_match)} relationships against {len(merged_entities)} entities")

        # Create entity mappings
        entity_by_name = {}  # name -> entity dict (case-insensitive)
        entity_by_id = {}  # id -> entity dict

        for entity in merged_entities:
            # Index by name (case-insensitive)
            name_key = entity["name"].lower()
            if name_key not in entity_by_name:
                entity_by_name[name_key] = entity

            # Index by ID
            entity_by_id[entity["id"]] = entity

        logger.info(f"Created mappings for {len(merged_entities)} entities")

        # Enrich relationship metadata, ensure entity1_name and entity2_name exist
        enriched_relationships = []
        for rel in relationships_to_match:
            enriched_rel = rel.copy()

            entity1_id = enriched_rel.get("entity1", "")
            entity2_id = enriched_rel.get("entity2", "")

            # Ensure entity1_name exists
            if "entity1_name" not in enriched_rel or not enriched_rel["entity1_name"]:
                if entity1_id and entity1_id in entity_by_id:
                    enriched_rel["entity1_name"] = entity_by_id[entity1_id]["name"]
                    logger.debug(f"Added entity1_name '{enriched_rel['entity1_name']}' for relationship with entity1 ID {entity1_id}")
                else:
                    logger.warning(f"Could not determine entity1_name for relationship: {rel}")
                    continue

            # Ensure entity2_name exists
            if "entity2_name" not in enriched_rel or not enriched_rel["entity2_name"]:
                if entity2_id and entity2_id in entity_by_id:
                    enriched_rel["entity2_name"] = entity_by_id[entity2_id]["name"]
                    logger.debug(f"Added entity2_name '{enriched_rel['entity2_name']}' for relationship with entity2 ID {entity2_id}")
                else:
                    logger.warning(f"Could not determine entity2_name for relationship: {rel}")
                    continue

            enriched_relationships.append(enriched_rel)

        logger.info(f"Enriched {len(enriched_relationships)} relationships with complete metadata")

        # Create relationship signatures for deduplication, considering entity pairs without direction
        unique_relationships = []
        relationship_pair_signatures = set()

        for rel in enriched_relationships:
            entity1_name = rel["entity1_name"].lower()
            entity2_name = rel["entity2_name"].lower()

            # Create direction-agnostic signature by sorting entity names
            entities = sorted([entity1_name, entity2_name])
            pair_signature = (entities[0], entities[1])

            # If this entity pair already has a relationship, skip
            if pair_signature in relationship_pair_signatures:
                continue

            relationship_pair_signatures.add(pair_signature)
            unique_relationships.append(rel)

        logger.info(f"Found {len(unique_relationships)} unique relationships after deduplication")

        return unique_relationships

    def _update_relationships(self, relationships: List[Dict[str, Any]],
                             entity_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update relationships with matched entities.

        Args:
            relationships: List of relationships
            entity_map: Mapping of entity IDs to entities

        Returns:
            List of updated relationships
        """
        updated_relationships = []

        for relationship in relationships:
            updated_relationship = relationship.copy()

            # Get entities
            entity1 = entity_map.get(relationship["entity1"])
            entity2 = entity_map.get(relationship["entity2"])

            # Check if both entities are verified
            if entity1 and entity2 and entity1.get("verified") and entity2.get("verified"):
                updated_relationship["verified"] = True
            else:
                updated_relationship["verified"] = False

            updated_relationships.append(updated_relationship)

        return updated_relationships

    def _merge_relationships(self, relationships: List[Dict[str, Any]], entity_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge relationships with matched entities.

        Args:
            relationships: List of relationships to merge
            entity_map: Mapping of entity IDs to entities

        Returns:
            List of merged relationships
        """
        merged_relationships = []

        for relationship in relationships:
            updated_relationship = relationship.copy()

            # Get entities
            entity1 = entity_map.get(relationship["entity1"])
            entity2 = entity_map.get(relationship["entity2"])

            # Check if both entities are verified
            if entity1 and entity2 and entity1.get("verified") and entity2.get("verified"):
                updated_relationship["verified"] = True
            else:
                updated_relationship["verified"] = False

            merged_relationships.append(updated_relationship)

        return merged_relationships

    def match(self, extraction_result: Dict[str, Any], ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Match extracted entities to text from OCR.

        Args:
            extraction_result: Output from EntityExtractor
            ocr_result: Output from OCR

        Returns:
            Dictionary with matched entities and relationships
        """
        logger.info(f"Matching entities and relationships from {extraction_result['filename']}")

        # 二次筛选实体，去掉名字、学校、网址、日期、IP地址、具体数值等
        # 这一步放在实体匹配前进行，确保不必要的实体在早期就被过滤掉
        original_entities = extraction_result.get("entities", [])
        filtered_entities = self._filter_entities(original_entities)
        extraction_result["entities"] = filtered_entities

        logger.info(f"二次筛选: 过滤前 {len(original_entities)} 个实体，过滤后 {len(filtered_entities)} 个实体")

        # 跟踪过滤前后的实体ID变化
        filter_id_map = self._track_entity_id_changes(original_entities, filtered_entities)

        # Get page text map from OCR result
        page_text_map = {}
        for page_no, page_data in ocr_result.get("pages", {}).items():
            if "text" in page_data:
                page_text_map[int(page_no)] = page_data["text"]

        # Match entities
        matched_entities = self._match_entities(filtered_entities, page_text_map)

        # 跟踪匹配前后的实体ID变化
        match_id_map = self._track_entity_id_changes(filtered_entities, matched_entities)

        # 合并两个ID映射
        combined_id_map = {**filter_id_map, **match_id_map}
        # 处理级联映射 (A->B, B->C => A->C)
        for old_id, new_id in filter_id_map.items():
            if new_id in match_id_map:
                combined_id_map[old_id] = match_id_map[new_id]

        # 确保每个关系中的实体引用使用正确的ID
        updated_relationships = []
        for rel in extraction_result.get("relationships", []):
            new_rel = rel.copy()

            # 更新entity1/entity2
            if "entity1" in new_rel and new_rel["entity1"] in combined_id_map:
                new_rel["entity1"] = combined_id_map[new_rel["entity1"]]
            if "entity2" in new_rel and new_rel["entity2"] in combined_id_map:
                new_rel["entity2"] = combined_id_map[new_rel["entity2"]]

            # 同时检查source/target字段
            if "source" in new_rel and new_rel["source"] in combined_id_map:
                new_rel["source"] = combined_id_map[new_rel["source"]]
            if "target" in new_rel and new_rel["target"] in combined_id_map:
                new_rel["target"] = combined_id_map[new_rel["target"]]

            # 只添加两个实体都存在于匹配后实体列表中的关系
            entity1_exists = False
            entity2_exists = False

            if "entity1" in new_rel:
                entity1_exists = any(e["id"] == new_rel["entity1"] for e in matched_entities)
            if "entity2" in new_rel:
                entity2_exists = any(e["id"] == new_rel["entity2"] for e in matched_entities)

            if entity1_exists and entity2_exists:
                updated_relationships.append(new_rel)
            else:
                logger.warning(f"关系中的一个或两个实体不存在，舍弃关系: {new_rel.get('entity1_name', '')} -> {new_rel.get('entity2_name', '')}")

        # Create entity map for consistency checks
        entity_map = {entity["id"]: entity for entity in matched_entities}

        # 处理孤立实体 - 确保每个实体至少有一个关系
        final_relationships = self._check_and_add_default_relationships(updated_relationships, matched_entities)

        # Create result dictionary
        result = {
            "filename": extraction_result["filename"],
            "entities": matched_entities,
            "relationships": final_relationships
        }

        # 当实体匹配后，重新生成所有ID
        self._regenerate_ids(result)

        return result

    def match_with_existing(self, extraction_result: Dict[str, Any],
                           ocr_result: Dict[str, Any],
                           existing_entities: List[Dict[str, Any]],
                           source_file: str) -> Dict[str, Any]:
        """
        Match extraction results with existing entities and relationships.

        Args:
            extraction_result: Output from EntityExtractor
            ocr_result: Output from OCR
            existing_entities: List of existing entities
            source_file: Source file name

        Returns:
            Dictionary with matched entities and relationships
        """
        logger.info(f"Matching entities and relationships from {extraction_result['filename']} with existing data")

        # 先进行二次筛选，过滤掉不需要的实体
        # 确保二次筛选在匹配和合并之前进行
        original_entities = extraction_result.get("entities", [])
        filtered_entities = self._filter_entities(original_entities)
        extraction_result["entities"] = filtered_entities
        existing_entities = self._filter_entities(existing_entities)

        logger.info(f"二次筛选: 新实体过滤前 {len(original_entities)} 个，过滤后 {len(filtered_entities)} 个")
        logger.info(f"二次筛选: 现有实体过滤前 {len(existing_entities)} 个，过滤后 {len(existing_entities)} 个")

        # 跟踪过滤前后的实体ID变化
        filter_id_map = self._track_entity_id_changes(original_entities, filtered_entities)

        # Get page text map from OCR result
        page_text_map = {}
        for page_no, page_data in ocr_result.get("pages", {}).items():
            if "text" in page_data:
                page_text_map[int(page_no)] = page_data["text"]

        # Match entities with text from OCR
        matched_entities = self._match_entities(filtered_entities, page_text_map)

        # 跟踪匹配前后的实体ID变化
        match_id_map = self._track_entity_id_changes(filtered_entities, matched_entities)

        # 合并两个ID映射
        combined_id_map = {**filter_id_map, **match_id_map}
        # 处理级联映射 (A->B, B->C => A->C)
        for old_id, new_id in filter_id_map.items():
            if new_id in match_id_map:
                combined_id_map[old_id] = match_id_map[new_id]

        # 确保每个关系中的实体引用使用正确的ID
        updated_relationships = []
        for rel in extraction_result.get("relationships", []):
            new_rel = rel.copy()

            # 更新entity1/entity2
            if "entity1" in new_rel and new_rel["entity1"] in combined_id_map:
                new_rel["entity1"] = combined_id_map[new_rel["entity1"]]
            if "entity2" in new_rel and new_rel["entity2"] in combined_id_map:
                new_rel["entity2"] = combined_id_map[new_rel["entity2"]]

            # 同时检查source/target字段
            if "source" in new_rel and new_rel["source"] in combined_id_map:
                new_rel["source"] = combined_id_map[new_rel["source"]]
            if "target" in new_rel and new_rel["target"] in combined_id_map:
                new_rel["target"] = combined_id_map[new_rel["target"]]

            # 只添加两个实体都存在于匹配后实体列表中的关系
            entity1_exists = False
            entity2_exists = False

            if "entity1" in new_rel:
                entity1_exists = any(e["id"] == new_rel["entity1"] for e in matched_entities)
            if "entity2" in new_rel:
                entity2_exists = any(e["id"] == new_rel["entity2"] for e in matched_entities)

            if entity1_exists and entity2_exists:
                updated_relationships.append(new_rel)
            else:
                logger.warning(f"关系中的一个或两个实体不存在，舍弃关系: {new_rel.get('entity1_name', '')} -> {new_rel.get('entity2_name', '')}")

        # 合并已匹配的实体与现有实体
        merged_entities = self._merge_entities(matched_entities, existing_entities, source_file)

        # 跟踪合并前后的实体ID变化
        merge_id_map = self._track_entity_id_changes(matched_entities, merged_entities)

        # 合并所有ID映射
        final_id_map = {**combined_id_map, **merge_id_map}
        for old_id, new_id in combined_id_map.items():
            if new_id in merge_id_map:
                final_id_map[old_id] = merge_id_map[new_id]

        # 更新关系中的实体引用，使用最终的ID映射
        final_relationships = []
        for rel in updated_relationships:
            final_rel = rel.copy()

            # 更新所有可能的实体引用字段
            for field in ["entity1", "entity2", "source", "target"]:
                if field in final_rel and final_rel[field] in final_id_map:
                    final_rel[field] = final_id_map[final_rel[field]]

            # 验证关系中的实体是否存在
            entity1_field = "entity1" if "entity1" in final_rel else "source"
            entity2_field = "entity2" if "entity2" in final_rel else "target"

            entity1_exists = any(e["id"] == final_rel.get(entity1_field) for e in merged_entities)
            entity2_exists = any(e["id"] == final_rel.get(entity2_field) for e in merged_entities)

            if entity1_exists and entity2_exists:
                final_relationships.append(final_rel)
            else:
                logger.warning(f"合并后关系中的一个或两个实体不存在，舍弃关系: {final_rel.get('entity1_name', '')} -> {final_rel.get('entity2_name', '')}")

        # 处理孤立实体 - 确保每个实体至少有一个关系
        final_relationships = self._check_and_add_default_relationships(final_relationships, merged_entities)

        # Create result dictionary
        result = {
            "entities": merged_entities,
            "relationships": final_relationships
        }

        # 最后重新生成所有ID，从e0001开始
        self._regenerate_ids(result)

        return result

    def _match_entities(self, entities: List[Dict[str, Any]], page_text_map: Dict[int, str]) -> List[Dict[str, Any]]:
        """
        Match entities against the original text.

        Args:
            entities: List of entities
            page_text_map: Mapping of page numbers to OCR text

        Returns:
            List of matched entities
        """
        matched_entities = []

        for entity in tqdm(entities, desc="Matching entities"):
            matched_entity = entity.copy()

            # Check if the entity appears in the text
            entity_name = entity["name"]
            entity_page = entity["page_number"]

            # Get the text for the entity's page
            page_text = page_text_map.get(entity_page, "")

            # Check if the entity name appears in the page text
            if self._entity_appears_in_text(entity_name, page_text):
                matched_entity["verified"] = True

                # Find all occurrences of the entity in all pages
                all_occurrences = self._find_all_occurrences(entity_name, page_text_map)
                matched_entity["all_occurrences"] = all_occurrences

                # Determine first occurrence
                if all_occurrences:
                    first_occurrence = min(all_occurrences)
                    matched_entity["first_occurrence_page"] = first_occurrence
                else:
                    matched_entity["first_occurrence_page"] = entity_page
            else:
                # Entity not found in the text
                matched_entity["verified"] = False
                matched_entity["all_occurrences"] = []
                matched_entity["first_occurrence_page"] = entity_page

            matched_entities.append(matched_entity)

        return matched_entities

    def _entity_appears_in_text(self, entity_name: str, text: str) -> bool:
        """
        Check if an entity appears in the text.

        Args:
            entity_name: Name of the entity
            text: Text to search in

        Returns:
            True if the entity appears in the text, False otherwise
        """
        # 处理空文本情况
        if not text or not entity_name:
            return False

        # 规范化实体名称和文本
        entity_name_norm = entity_name.lower().strip()
        text_norm = text.lower()

        # 简单字符串匹配 - 更宽松的方式
        if entity_name_norm in text_norm:
            return True

        # 尝试去除常见标点符号的匹配
        clean_entity_name = re.sub(r'[.,;:"\'\-\(\)]', '', entity_name_norm)
        clean_text = re.sub(r'[.,;:"\'\-\(\)]', '', text_norm)
        if clean_entity_name in clean_text:
            return True

        # 单词边界匹配 - 更严格的方式，但可能有误报
        try:
            pattern = r'\b' + re.escape(entity_name_norm) + r'\b'
            if re.search(pattern, text_norm, re.IGNORECASE):
                return True
        except re.error:
            # 防止正则表达式错误
            pass

        # 处理缩写和全称的特殊情况
        abbreviations = {
            "tcp": "transmission control protocol",
            "ip": "internet protocol",
            "udp": "user datagram protocol",
            "http": "hypertext transfer protocol",
            "dns": "domain name system",
            "ftp": "file transfer protocol",
            "smtp": "simple mail transfer protocol",
            "osi": "open systems interconnection",
            "lan": "local area network",
            "wan": "wide area network",
            "vlan": "virtual local area network",
            "mac": "media access control",
            "rip": "routing information protocol",
            "ospf": "open shortest path first",
            "bgp": "border gateway protocol",
            "dhcp": "dynamic host configuration protocol"
        }

        # 如果实体是缩写，检查全称是否在文本中
        if entity_name_norm.lower() in abbreviations:
            full_form = abbreviations[entity_name_norm.lower()]
            if full_form in text_norm:
                return True

        # 如果实体是全称，检查缩写是否在文本中
        for abbr, full_form in abbreviations.items():
            if entity_name_norm == full_form and abbr in text_norm.split():
                return True

        # 处理单复数形式
        if entity_name_norm.endswith('s') and entity_name_norm[:-1] in text_norm:
            return True
        if not entity_name_norm.endswith('s') and entity_name_norm + 's' in text_norm:
            return True

        # 最后，尝试更灵活的匹配 - 分词匹配
        entity_words = entity_name_norm.split()
        if len(entity_words) > 1:
            # 对于多词实体，如果所有单词都在文本中且相对位置正确，认为匹配
            text_words = text_norm.split()
            for i in range(len(text_words) - len(entity_words) + 1):
                match = True
                for j, word in enumerate(entity_words):
                    if word != text_words[i + j]:
                        match = False
                        break
                if match:
                    return True

        # 未找到匹配
        return False

    def _find_all_occurrences(self, entity_name: str, page_text_map: Dict[int, str]) -> List[int]:
        """
        Find all occurrences of an entity in all pages.

        Args:
            entity_name: Name of the entity
            page_text_map: Mapping of page numbers to OCR text

        Returns:
            List of page numbers where the entity appears
        """
        occurrences = []

        # Normalize entity name
        entity_name_norm = entity_name.lower()

        # Check each page
        for page_num, page_text in page_text_map.items():
            # Normalize page text
            page_text_norm = page_text.lower()

            # Check for exact match
            if entity_name_norm in page_text_norm:
                occurrences.append(page_num)
                continue

            # Check for match with different case
            pattern = r'\b' + re.escape(entity_name_norm) + r'\b'
            if re.search(pattern, page_text_norm, re.IGNORECASE):
                occurrences.append(page_num)
                continue

            # Check for common abbreviations
            if len(entity_name_norm) > 3 and entity_name_norm.isupper():
                # For abbreviations, check if the full form is in the text
                abbreviations = {
                    "tcp": "transmission control protocol",
                    "ip": "internet protocol",
                    "udp": "user datagram protocol",
                    "http": "hypertext transfer protocol",
                    "dns": "domain name system",
                    "ftp": "file transfer protocol",
                    "smtp": "simple mail transfer protocol",
                    "osi": "open systems interconnection",
                    "lan": "local area network",
                    "wan": "wide area network",
                    "vlan": "virtual local area network",
                    "mac": "media access control",
                    "rip": "routing information protocol",
                    "ospf": "open shortest path first",
                    "bgp": "border gateway protocol",
                    "dhcp": "dynamic host configuration protocol"
                }

                if entity_name_norm.lower() in abbreviations:
                    full_form = abbreviations[entity_name_norm.lower()]
                    if full_form in page_text_norm:
                        occurrences.append(page_num)

        return occurrences

    def _merge_entities(self, entities: List[Dict[str, Any]], existing_entities: List[Dict[str, Any]], source_file: str) -> List[Dict[str, Any]]:
        """
        Merge new entities with existing entities and verify against OCR text.

        Args:
            entities: List of new entities to merge
            existing_entities: List of existing entities to merge against
            source_file: Source file name

        Returns:
            List of merged entities
        """
        logger.info(f"Merging {len(entities)} new entities against {len(existing_entities)} existing entities")

        merged_entities = existing_entities.copy()

        # Map of entity names to indices in the merged list for fast lookup
        entity_name_map = {}
        for idx, entity in enumerate(merged_entities):
            name = entity["name"].lower()
            if name not in entity_name_map:
                entity_name_map[name] = []
            entity_name_map[name].append(idx)

        # For each new entity, check if it already exists and merge if necessary
        for new_entity in entities:
            new_name = new_entity["name"].lower()

            # Check if an entity with this name already exists
            if new_name in entity_name_map:
                # Entity exists, update its occurrences
                for idx in entity_name_map[new_name]:
                    existing_entity = merged_entities[idx]

                    # Merge occurrences
                    if "occurrences" in new_entity and "occurrences" in existing_entity:
                        existing_occurrences = {tuple(occ) for occ in existing_entity["occurrences"]}
                        for occ in new_entity["occurrences"]:
                            existing_occurrences.add(tuple(occ))
                        existing_entity["occurrences"] = [list(occ) for occ in existing_occurrences]

                    # Update first_occurrence flag if needed
                    if new_entity.get("first_occurrence", False) and not existing_entity.get("first_occurrence", False):
                        existing_entity["first_occurrence"] = True

                    # Merge other fields as needed (e.g., definition, type)
                    if "definition" in new_entity and "definition" not in existing_entity:
                        existing_entity["definition"] = new_entity["definition"]

                    if "type" in new_entity and new_entity["type"] != "other" and (
                            "type" not in existing_entity or existing_entity["type"] == "other"):
                        existing_entity["type"] = new_entity["type"]
            else:
                # New entity, add it to the merged list
                merged_entities.append(new_entity)

                # Update the name map
                if new_name not in entity_name_map:
                    entity_name_map[new_name] = []
                entity_name_map[new_name].append(len(merged_entities) - 1)

        logger.info(f"Merged entities: {len(merged_entities)}")
        return merged_entities

    def _regenerate_ids(self, result: Dict[str, Any]) -> None:
        """
        重新生成实体和关系的ID，确保唯一性

        Args:
            result: 结果字典，包含实体和关系列表
        """
        if "entities" not in result or "relationships" not in result:
            return

        # 创建旧实体ID到新实体ID的映射
        id_map = {}

        # 为实体生成新ID
        for i, entity in enumerate(result["entities"]):
            if "id" in entity:
                old_id = entity["id"]
                new_id = f"e{i+1:04d}"  # 格式: e0001, e0002, ...
                id_map[old_id] = new_id
                entity["id"] = new_id

        # 更新关系中的实体引用
        for rel in result["relationships"]:
            if "entity1" in rel and rel["entity1"] in id_map:
                rel["entity1"] = id_map[rel["entity1"]]
            if "entity2" in rel and rel["entity2"] in id_map:
                rel["entity2"] = id_map[rel["entity2"]]

            # 同时检查并更新可能存在的source和target字段
            if "source" in rel and rel["source"] in id_map:
                rel["source"] = id_map[rel["source"]]
            if "target" in rel and rel["target"] in id_map:
                rel["target"] = id_map[rel["target"]]

        # 为关系生成新ID
        for i, rel in enumerate(result["relationships"]):
            if "id" in rel:
                rel["id"] = f"r{i+1:04d}"  # 格式: r0001, r0002, ...

        logger.info(f"重新生成ID: {len(result['entities'])} 个实体 (e0001-e{len(result['entities']):04d})，" +
                   f"{len(result['relationships'])} 个关系 (r0001-r{len(result['relationships']):04d})")

        # 记录新ID分配情况
        if result["entities"]:
            first_entity = result["entities"][0]
            last_entity = result["entities"][-1]
            logger.info(f"实体ID范围: {first_entity['id']} ({first_entity['name']}) - {last_entity['id']} ({last_entity['name']})")

        if result["relationships"]:
            first_rel = result["relationships"][0]
            last_rel = result["relationships"][-1]
            logger.info(f"关系ID范围: {first_rel['id']} ({first_rel.get('entity1_name', '')} -> {first_rel.get('entity2_name', '')}) - " +
                       f"{last_rel['id']} ({last_rel.get('entity1_name', '')} -> {last_rel.get('entity2_name', '')})")

        return

    def _check_and_add_default_relationships(self, relationships: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        检查并为孤立实体添加默认关系，将孤立实体优先与同一页面的实体关联

        Args:
            relationships: 关系列表
            entities: 实体列表

        Returns:
            更新后的关系列表
        """
        # 统计每个实体的关系数量
        entity_relationship_count = {}

        # 初始化计数
        for entity in entities:
            entity_relationship_count[entity["id"]] = 0

        # 计算每个实体的关系数量
        for rel in relationships:
            entity_relationship_count[rel["entity1"]] = entity_relationship_count.get(rel["entity1"], 0) + 1
            entity_relationship_count[rel["entity2"]] = entity_relationship_count.get(rel["entity2"], 0) + 1

        # 找出孤立实体
        isolated_entities = [entity for entity in entities if entity_relationship_count.get(entity["id"], 0) == 0]

        if not isolated_entities:
            logger.info("没有发现孤立实体，所有实体都有关系")
            return relationships

        logger.info(f"发现 {len(isolated_entities)} 个孤立实体，添加默认关系")

        # 按页面对实体分组
        entities_by_page = {}
        for entity in entities:
            if "page_number" in entity:
                page_num = entity["page_number"]
                if page_num not in entities_by_page:
                    entities_by_page[page_num] = []
                entities_by_page[page_num].append(entity)

        # 为孤立实体添加关系
        for isolated_entity in isolated_entities:
            relationship_added = False

            # 1. 优先与同一页面的实体建立关系
            if "page_number" in isolated_entity and isolated_entity["page_number"] in entities_by_page:
                page_entities = entities_by_page[isolated_entity["page_number"]]
                same_page_entities = [e for e in page_entities if e["id"] != isolated_entity["id"]]

                if same_page_entities:
                    partner_entity = same_page_entities[0]  # 取同一页面的第一个实体
                    relationship = {
                        "entity1": isolated_entity["id"],
                        "entity2": partner_entity["id"],
                        "entity1_name": isolated_entity["name"],
                        "entity2_name": partner_entity["name"],
                        "type": "related_to",
                        "verified": False
                    }
                    if "page_number" in isolated_entity:
                        relationship["page_number"] = isolated_entity["page_number"]
                    if "chapter" in isolated_entity:
                        relationship["chapter"] = isolated_entity["chapter"]

                    relationships.append(relationship)
                    logger.info(f"为孤立实体 '{isolated_entity['name']}' 添加与同页面实体 '{partner_entity['name']}' 的关系，页码：{isolated_entity.get('page_number', '未知')}")
                    relationship_added = True

            # 2. 如果没有同页面实体，尝试与任意非孤立实体建立关系
            if not relationship_added:
                non_isolated_entities = [e for e in entities if e["id"] != isolated_entity["id"] and entity_relationship_count.get(e["id"], 0) > 0]
                if non_isolated_entities:
                    partner_entity = non_isolated_entities[0]
                    relationship = {
                        "entity1": isolated_entity["id"],
                        "entity2": partner_entity["id"],
                        "entity1_name": isolated_entity["name"],
                        "entity2_name": partner_entity["name"],
                        "type": "related_to",
                        "verified": False
                    }
                    if "page_number" in isolated_entity:
                        relationship["page_number"] = isolated_entity["page_number"]
                    if "chapter" in isolated_entity:
                        relationship["chapter"] = isolated_entity["chapter"]

                    relationships.append(relationship)
                    logger.info(f"为孤立实体 '{isolated_entity['name']}' 添加与非孤立实体 '{partner_entity['name']}' 的关系")
                    relationship_added = True

            # 3. 最后尝试与任意实体建立关系
            if not relationship_added:
                other_entities = [e for e in entities if e["id"] != isolated_entity["id"]]
                if other_entities:
                    partner_entity = other_entities[0]
                    relationship = {
                        "entity1": isolated_entity["id"],
                        "entity2": partner_entity["id"],
                        "entity1_name": isolated_entity["name"],
                        "entity2_name": partner_entity["name"],
                        "type": "related_to",
                        "verified": False
                    }
                    if "page_number" in isolated_entity:
                        relationship["page_number"] = isolated_entity["page_number"]
                    if "chapter" in isolated_entity:
                        relationship["chapter"] = isolated_entity["chapter"]

                    relationships.append(relationship)
                    logger.info(f"为孤立实体 '{isolated_entity['name']}' 添加与任意实体 '{partner_entity['name']}' 的关系")

        return relationships

    def _track_entity_id_changes(self, old_entities: List[Dict[str, Any]], new_entities: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        跟踪实体ID的变化，用于更新关系引用

        Args:
            old_entities: 旧实体列表
            new_entities: 新实体列表

        Returns:
            旧ID到新ID的映射字典
        """
        id_map = {}
        name_to_new_id = {}

        # 创建名称到新ID的映射
        for entity in new_entities:
            name_to_new_id[entity["name"].lower()] = entity["id"]

        # 映射旧ID到新ID
        for old_entity in old_entities:
            old_id = old_entity["id"]
            name = old_entity["name"].lower()
            if name in name_to_new_id:
                id_map[old_id] = name_to_new_id[name]
                logger.debug(f"映射临时ID {old_id} -> {name_to_new_id[name]} (实体: {name})")

        # 记录映射统计
        logger.info(f"实体ID映射: 共追踪 {len(id_map)} 个ID变更")

        return id_map
