"""
Type explorer for discovering new entity types and relationship types.
"""

import os
import json
import logging
import math
from typing import Dict, List, Any, Set
from collections import Counter

from tqdm import tqdm

import config
from utils.llm_utils import LLMClient

logger = logging.getLogger(__name__)

class TypeExplorer:
    """
    Class for exploring and discovering new entity types and relationship types.
    """

    def __init__(self, model_name: str = config.DEFAULT_MODEL, api_key: str = None, api_base: str = None):
        """
        Initialize the type explorer.

        Args:
            model_name: Name of the LLM model to use
            api_key: API key for the LLM service
            api_base: API base URL for the LLM service
        """
        self.model_name = model_name
        self.api_key = api_key or config.OPENAI_API_KEY
        self.api_base = api_base or config.OPENAI_API_BASE

        # Initialize LLM client
        self.llm_client = LLMClient(model_name, api_key, api_base)

        logger.info(f"Initialized type explorer with model {model_name}")

        # Load exploration prompt templates
        self.entity_type_exploration_prompt = self._load_prompt_template("entity_type_exploration")
        self.relationship_type_exploration_prompt = self._load_prompt_template("relationship_type_exploration")

    def explore(self, definition_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explore and discover new entity types and relationship types.

        Args:
            definition_result: Dictionary containing defined entities and relationships

        Returns:
            Dictionary containing exploration results
        """
        logger.info(f"Exploring entity and relationship types from {definition_result['filename']}")

        result = {
            "filename": definition_result.get("filename", "unknown"),
            "chapter": definition_result.get("chapter", ""),
            "entities": [],
            "relationships": [],
            "new_entity_types": [],
            "new_relationship_types": []
        }

        # 直接使用当前类型，规范化阶段已经更新了关系类型
        entities = definition_result.get("entities", [])
        relationships = definition_result.get("relationships", [])

        # Filter entities with type "other" or "concept"
        entities_to_explore = [
            e for e in entities
            if e.get("type", "").lower() in ["other", "concept"]
        ]
        logger.info(f"Found {len(entities_to_explore)} entities to explore")

        # Filter relationships with type "related_to"
        relationships_to_explore = [
            r for r in relationships
            if r.get("type", "").lower() == "related_to"
        ]
        logger.info(f"Found {len(relationships_to_explore)} relationships to explore")

        # Discover new entity types
        new_entity_types = self._discover_new_entity_types_from_others(entities_to_explore)

        # Evaluate and add quality metrics to new entity types
        if new_entity_types:
            logger.info("Evaluating quality of new entity types...")
            evaluated_entity_types = self._evaluate_entity_types_quality(new_entity_types, entities)

            # Store type IDs that need to be rejected due to low quality
            self.rejected_entity_types = {t["name"] for t in evaluated_entity_types
                                         if t.get("quality_metrics", {}).get("overall", 0) < 0.6}
            if self.rejected_entity_types:
                logger.warning(f"Rejecting {len(self.rejected_entity_types)} entity types due to low quality scores: {self.rejected_entity_types}")

            # Filter out low-quality types
            quality_entity_types = [t for t in evaluated_entity_types
                                   if t.get("quality_metrics", {}).get("overall", 0) >= 0.6]
            result["new_entity_types"] = quality_entity_types
            logger.info(f"Discovered {len(new_entity_types)} entity types, {len(quality_entity_types)} passed quality threshold")
        else:
            result["new_entity_types"] = []
            self.rejected_entity_types = set()

        # Discover new relationship types
        new_relationship_types = self._discover_new_relationship_types_from_others(relationships_to_explore)

        # Evaluate and add quality metrics to new relationship types
        if new_relationship_types:
            logger.info("Evaluating quality of new relationship types...")
            evaluated_rel_types = self._evaluate_relationship_types_quality(new_relationship_types, entities)

            # Store type IDs that need to be rejected due to low quality
            self.rejected_relationship_types = {t["name"] for t in evaluated_rel_types
                                              if t.get("quality_metrics", {}).get("overall", 0) < 0.6}
            if self.rejected_relationship_types:
                logger.warning(f"Rejecting {len(self.rejected_relationship_types)} relationship types due to low quality scores: {self.rejected_relationship_types}")

            # Filter out low-quality types
            quality_rel_types = [t for t in evaluated_rel_types
                                if t.get("quality_metrics", {}).get("overall", 0) >= 0.6]
            result["new_relationship_types"] = quality_rel_types
            logger.info(f"Discovered {len(new_relationship_types)} relationship types, {len(quality_rel_types)} passed quality threshold")
        else:
            result["new_relationship_types"] = []
            self.rejected_relationship_types = set()

        # Update entity types
        updated_entities = self._update_entity_types(entities, result["new_entity_types"])
        result["entities"] = updated_entities

        # Update relationship types
        updated_relationships = self._update_relationship_types(relationships, result["new_relationship_types"])
        result["relationships"] = updated_relationships

        # 保存探索结果到文件
        self._save_exploration_output(result)

        return result

    def _save_exploration_output(self, result: Dict[str, Any]) -> None:
        """
        保存探索结果到输出目录。

        Args:
            result: 包含探索结果的字典
        """
        # 确保输出目录存在
        output_dir = config.EXPLORATION_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        # 保存探索结果文件 - 包含所有实体和关系的完整结果
        combined_path = os.path.join(output_dir, "combined_explored.json")
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"探索结果已保存到 {combined_path}")

        # 保存新发现的类型文件 - 只包含新发现的实体类型和关系类型
        types_output = {
            "new_entity_types": result.get("new_entity_types", []),
            "new_relationship_types": result.get("new_relationship_types", [])
        }

        combined_types_path = os.path.join(output_dir, "combined_new_types.json")
        with open(combined_types_path, 'w', encoding='utf-8') as f:
            json.dump(types_output, f, indent=2, ensure_ascii=False)

        logger.info(f"新发现的类型已保存到 {combined_types_path}")

    def _discover_new_entity_types_from_others(self, other_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Discover new entity types from entities classified as "other".

        Args:
            other_entities: List of entities with type "other"

        Returns:
            List of new entity types
        """
        if not other_entities:
            logger.info("No 'other' entities found to explore")
            return []

        # Group entities by similar definitions and characteristics
        # Prepare data for LLM with names and definitions
        entity_examples = []
        for entity in other_entities:
            entity_examples.append({
                "id": entity.get("id", ""),
                "name": entity.get("name", ""),
                "definition": entity.get("definition", "")
            })

        # Get standard types in lowercase for comparison
        standard_types_lower = [t.lower() for t in config.ENTITY_TYPES]

        # Call LLM to analyze and suggest new types
        prompt = self.entity_type_exploration_prompt.format(
            entity_examples=json.dumps(entity_examples, indent=2),
            standard_types=", ".join(config.ENTITY_TYPES)
        )

        logger.info("Calling LLM to discover new entity types")
        response = self.llm_client.call(prompt)

        # Parse response to get new entity types
        try:
            # Try to parse JSON response
            parsed_response = self.llm_client.parse_list_response(response)

            # If we got a list of items directly, use that
            new_types = []
            if isinstance(parsed_response, list):
                new_types = parsed_response
            # If we got a dict with a "new_types" key, extract that
            elif isinstance(parsed_response, dict) and "new_types" in parsed_response:
                new_types = parsed_response.get("new_types", [])

            # Validate each new type has required fields and is not duplicate of standard types
            validated_types = []
            for new_type in new_types:
                # Skip if missing required fields
                if not all(key in new_type for key in ["name", "definition", "examples"]):
                    logger.warning(f"Skipping invalid new type missing required fields: {new_type}")
                    continue

                # Skip if type name is similar to a standard type
                type_name = new_type["name"].lower()
                if any(self._are_type_names_similar(type_name, std_type) for std_type in standard_types_lower):
                    logger.warning(f"Skipping new type '{new_type['name']}' that is similar to a standard type")
                    continue

                # Skip if already added a similar type
                if any(self._are_type_names_similar(type_name, validated.get("name", "").lower()) for validated in validated_types):
                    logger.warning(f"Skipping duplicate new type: {new_type['name']}")
                    continue

                validated_types.append(new_type)
                logger.info(f"Added new entity type: {new_type['name']}")

            return validated_types
        except Exception as e:
            logger.error(f"Error parsing new entity types: {e}")
            return []

    def _discover_new_relationship_types_from_others(self, other_relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Discover new relationship types from relationships classified as "other".

        Args:
            other_relationships: List of relationships with type "other"

        Returns:
            List of new relationship types
        """
        if not other_relationships:
            logger.info("No 'other' relationships found to explore")
            return []

        # Prepare data for LLM with source, target and definitions
        relationship_examples = []
        for relationship in other_relationships:
            relationship_examples.append({
                "id": relationship.get("id", ""),
                "entity1": relationship.get("entity1_name", ""),
                "entity2": relationship.get("entity2_name", ""),
                "source": relationship.get("source_name", ""),
                "target": relationship.get("target_name", ""),
                "definition": relationship.get("definition", "")
            })

        # Get standard types in lowercase for comparison
        standard_types_lower = [t.lower() for t in config.RELATIONSHIP_TYPES]

        # Call LLM to analyze and suggest new types
        prompt = self.relationship_type_exploration_prompt.format(
            relationship_examples=json.dumps(relationship_examples, indent=2),
            standard_types=", ".join(config.RELATIONSHIP_TYPES)
        )

        logger.info("Calling LLM to discover new relationship types")
        response = self.llm_client.call(prompt)

        # Dictionary to store mapping of similar types to standard types
        # This will be used later to update relationship types
        self.similar_to_standard_type_map = {}

        # Parse response to get new relationship types
        try:
            # Try to parse JSON response
            parsed_response = self.llm_client.parse_list_response(response)

            # If we got a list of items directly, use that
            new_types = []
            if isinstance(parsed_response, list):
                new_types = parsed_response
            # If we got a dict with a "new_types" key, extract that
            elif isinstance(parsed_response, dict) and "new_types" in parsed_response:
                new_types = parsed_response.get("new_types", [])

            # Validate each new type has required fields and is not duplicate of standard types
            validated_types = []
            for new_type in new_types:
                # Skip if missing required fields
                if not all(key in new_type for key in ["name", "definition", "examples"]):
                    logger.warning(f"Skipping invalid new type missing required fields: {new_type}")
                    continue

                # Check if type name is similar to a standard type
                type_name = new_type["name"].lower()
                similar_to_standard = False

                for std_type_lower in standard_types_lower:
                    if self._are_type_names_similar(type_name, std_type_lower):
                        # Find the original case of the standard type
                        original_std_type = next(
                            (t for t in config.RELATIONSHIP_TYPES if t.lower() == std_type_lower),
                            std_type_lower
                        )
                        logger.warning(f"New type '{new_type['name']}' is similar to standard type '{original_std_type}'")

                        # Record this mapping for later use in update_relationship_types
                        self.similar_to_standard_type_map[new_type["name"]] = original_std_type

                        # Store relationship IDs for updating to standard type
                        for example in new_type.get("examples", []):
                            rel_id = example.get("id")
                            if rel_id:
                                # If we have multiple examples with the same ID but different standard types,
                                # we'll use the last one (could be improved with more sophisticated logic)
                                if rel_id not in getattr(self, 'relationship_to_standard_type_map', {}):
                                    if not hasattr(self, 'relationship_to_standard_type_map'):
                                        self.relationship_to_standard_type_map = {}
                                    self.relationship_to_standard_type_map[rel_id] = original_std_type

                        similar_to_standard = True
                        break

                # Skip if it's similar to a standard type
                if similar_to_standard:
                    continue

                # Skip if already added a similar type
                if any(self._are_type_names_similar(type_name, validated.get("name", "").lower()) for validated in validated_types):
                    logger.warning(f"Skipping duplicate new type: {new_type['name']}")
                    continue

                validated_types.append(new_type)
                logger.info(f"Added new relationship type: {new_type['name']}")

            return validated_types
        except Exception as e:
            logger.error(f"Error parsing new relationship types: {e}")
            return []

    def _are_type_names_similar(self, name1: str, name2: str) -> bool:
        """
        Check if two type names are similar, ignoring case, spacing and common word variations.

        Args:
            name1: First type name
            name2: Second type name

        Returns:
            True if the names are similar, False otherwise
        """
        # Convert to lowercase and remove underscores, hyphens, and spaces
        name1 = name1.lower().replace('_', '').replace('-', '').replace(' ', '')
        name2 = name2.lower().replace('_', '').replace('-', '').replace(' ', '')

        # Exact match after normalization
        if name1 == name2:
            return True

        # Common word variations
        if name1 + 'of' == name2 or name2 + 'of' == name1:  # e.g., "part" vs "partof"
            return True

        # For "belongto" vs "isa" or "ispartof"
        belongs_variations = ["belongto", "belongsto", "ispartof"]
        isa_variations = ["isa", "isatypeof", "isasubclassof", "isrelatedto"]

        if name1 in belongs_variations and name2 in belongs_variations:
            return True
        if name1 in isa_variations and name2 in isa_variations:
            return True
        if name1 in belongs_variations and name2 in isa_variations:
            return True
        if name1 in isa_variations and name2 in belongs_variations:
            return True

        # Check for prefix/suffix relationships with >70% character overlap
        if len(name1) > len(name2):
            longer, shorter = name1, name2
        else:
            longer, shorter = name2, name1

        if shorter in longer and len(shorter) / len(longer) > 0.7:
            return True

        return False

    def _update_entity_types(self, entities: List[Dict[str, Any]], new_entity_types: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update entity types based on discovered new types.

        Args:
            entities: List of all entities
            new_entity_types: List of new entity types

        Returns:
            List of updated entities
        """
        updated_entities = []

        # Create a mapping of entity examples to their new types
        entity_type_mapping = {}
        for new_type in new_entity_types:
            for example in new_type.get("examples", []):
                entity_id = example.get("id")
                if entity_id:
                    entity_type_mapping[entity_id] = new_type["name"]

        # Track how many entities were updated
        update_counts = {"updated": 0, "skipped_rejected": 0}

        # Update entity types
        for entity in entities:
            updated_entity = entity.copy()
            entity_id = entity.get("id", "")
            old_type = entity.get("type", "").lower()

            # Check if this entity is in our mapping and is either "other" or "concept"
            if (entity_id in entity_type_mapping and old_type in ["other", "concept"]):
                new_type = entity_type_mapping[entity_id]

                # Skip if this type was rejected due to low quality
                if hasattr(self, 'rejected_entity_types') and new_type in self.rejected_entity_types:
                    logger.info(f"保持实体 '{entity.get('name')}' 类型为 '{old_type}'，因为新类型 '{new_type}' 未通过质量评估")
                    update_counts["skipped_rejected"] += 1
                else:
                    # Update to new type
                    updated_entity["type"] = new_type
                    logger.info(f"更新实体 '{entity.get('name')}' 类型: '{old_type}' -> '{new_type}'")
                    update_counts["updated"] += 1

            updated_entities.append(updated_entity)

        logger.info(f"更新了 {update_counts['updated']} 个实体的类型，跳过了 {update_counts['skipped_rejected']} 个实体（因质量低的类型）")

        return updated_entities

    def _update_relationship_types(self, relationships: List[Dict[str, Any]], new_relationship_types: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update relationship types based on discovered new types.

        Args:
            relationships: List of all relationships
            new_relationship_types: List of new relationship types

        Returns:
            List of updated relationships
        """
        updated_relationships = []

        # Create a mapping of relationship examples to their new types
        relationship_type_mapping = {}

        # First, add new types from the exploration
        for new_type in new_relationship_types:
            for example in new_type.get("examples", []):
                relationship_id = example.get("id")
                if relationship_id:
                    relationship_type_mapping[relationship_id] = new_type["name"]

        # Next, add mappings to standard types we discovered during the exploration
        if hasattr(self, 'relationship_to_standard_type_map'):
            for rel_id, std_type in self.relationship_to_standard_type_map.items():
                relationship_type_mapping[rel_id] = std_type
                logger.info(f"Mapping relationship {rel_id} to standard type '{std_type}'")

        # Count how many relationships were updated
        update_count = {"to_new_types": 0, "to_standard_types": 0, "skipped_rejected": 0}

        # Update relationship types
        for relationship in relationships:
            updated_relationship = relationship.copy()
            rel_id = relationship.get("id")
            old_type = relationship.get("type", "related_to")

            # Check if this relationship should be updated
            if rel_id in relationship_type_mapping:
                new_type = relationship_type_mapping[rel_id]

                # Only update if it's currently "related_to" or it was a standard type that was
                # incorrectly identified (and is now being corrected to another standard type)
                if old_type.lower() == "related_to":
                    # Skip if this type was rejected due to low quality
                    if hasattr(self, 'rejected_relationship_types') and new_type in self.rejected_relationship_types:
                        logger.info(f"保持关系 {rel_id} 类型为 '{old_type}'，因为新类型 '{new_type}' 未通过质量评估")
                        update_count["skipped_rejected"] += 1
                    else:
                        # Update to new type
                        updated_relationship["type"] = new_type

                        # Check if updating to a standard or new type
                        if new_type in config.RELATIONSHIP_TYPES:
                            update_count["to_standard_types"] += 1
                        else:
                            update_count["to_new_types"] += 1

                        logger.info(f"更新关系 {rel_id} 类型: '{old_type}' -> '{new_type}'")

            updated_relationships.append(updated_relationship)

        # Log summary of updates
        logger.info(f"更新了 {update_count['to_new_types']} 个关系到新类型")
        logger.info(f"更新了 {update_count['to_standard_types']} 个关系到标准类型")
        logger.info(f"跳过了 {update_count['skipped_rejected']} 个关系（因质量低的类型）")

        return updated_relationships

    def _parse_new_types(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse new types from LLM response.

        Args:
            response: LLM response

        Returns:
            List of new types
        """
        try:
            # Try to parse JSON response
            parsed_response = self.llm_client.parse_list_response(response)

            # 增加日志，显示解析后的响应结构
            logger.info(f"Parsed response type: {type(parsed_response).__name__}")
            if isinstance(parsed_response, dict):
                logger.info(f"Response keys: {list(parsed_response.keys())}")

            # If we got a list of items directly, use that
            new_types = []
            if isinstance(parsed_response, list):
                new_types = parsed_response
            # If we got a dict with a "new_types" key, extract that
            elif isinstance(parsed_response, dict) and "new_types" in parsed_response:
                new_types = parsed_response.get("new_types", [])

            # Validate each new type has required fields
            validated_types = []
            for new_type in new_types:
                if "name" in new_type and "definition" in new_type and "examples" in new_type:
                    validated_types.append(new_type)
                    logger.info(f"Validated new type: {new_type['name']}")
                else:
                    logger.warning(f"Skipping invalid new type: {new_type}")

            return validated_types
        except Exception as e:
            logger.error(f"Error parsing new types: {e}")
            # Return empty list on error
            return []

    def _load_prompt_template(self, template_name: str) -> str:
        """
        Load a prompt template.

        Args:
            template_name: Name of the template

        Returns:
            Prompt template string
        """
        if template_name == "entity_type_exploration":
            return """
            Analyze the following entities from a computer network knowledge graph.
            These entities are currently classified as "other" or "concept" and may represent new meaningful categories.

            Entity Examples:
            {entity_examples}

            Standard Entity Types: {standard_types}

            Your task is to:
            1. Analyze these entities to identify patterns that suggest new meaningful category types
            2. ONLY suggest new entity types that are NOT already covered by the standard types - avoid suggesting any variation or synonym of the standard types
            3. Consider the context and relationships of these entities in the computer network domain

            IMPORTANT:
            - Do NOT suggest types that are similar to standard types (like suggesting "NetworkDevice" if "device" is standard)
            - Each type should be COMPLETELY NEW and DISTINCT from the standard types listed above
            - Use PascalCase for type names (e.g., NetworkTopology, FrequencyBand, SecurityProtocol)
            - Focus on meaningful groupings that would help students understand computer networks

            For each suggested new type, provide:
            - A name (using PascalCase, e.g., NetworkTopology, FrequencyBand)
            - A clear definition that explains why this is a distinct category
            - Examples of entities that should be classified with this new type

            Return your analysis in the following JSON format:
            {{
                "new_types": [
                    {{
                        "name": "TypeName",
                        "definition": "Type definition",
                        "examples": [list of entity objects that match this type]
                    }}
                ]
            }}

            If you don't recommend any new types, return an empty list for "new_types".
            """
        elif template_name == "relationship_type_exploration":
            return """
            Analyze the following relationships from a computer network knowledge graph.
            These relationships are currently classified as "related_to" and may represent more specific relationship types.

            Relationship Examples:
            {relationship_examples}

            Standard Relationship Types: {standard_types}

            EDUCATIONAL CONTEXT:
            This knowledge graph is designed to help students learn computer networks. The relationships should
            help students understand how different concepts are connected and the best order to learn them.

            Your task is to:
            1. Analyze these relationships to identify patterns that suggest new meaningful relationship types
            2. ONLY suggest relationship types that are COMPLETELY DIFFERENT from the standard types listed above
            3. DO NOT suggest variations, synonyms or case changes of standard types (e.g., do not suggest "PartOf" if "part_of" is standard)
            4. Consider the educational value of these relationships - how they can help guide learning

            IMPORTANT CONSTRAINTS:
            - Each suggested type MUST be distinct from ALL standard types - avoid any overlap in meaning
            - Use PascalCase for type names (e.g., Implements, Transmits, ModelsProcess)
            - DO NOT create types that can be represented by existing standard types
            - Focus on educational relationships that would help students understand the connections

            For each suggested new type, provide:
            - A name (using PascalCase, e.g., Implements, Transmits, ModelsProcess)
            - A clear definition that explains its educational significance and how it differs from standard types
            - Examples of relationships that should be classified with this new type

            Return your analysis in the following JSON format:
            {{
                "new_types": [
                    {{
                        "name": "TypeName",
                        "definition": "Type definition with educational relevance",
                        "examples": [list of relationship objects that match this type]
                    }}
                ]
            }}

            If you don't recommend any new types, return an empty list for "new_types".
            """
        else:
            return ""

    def _evaluate_entity_types_quality(self, entity_types: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        评估新发现的实体类型的质量：置信度、清晰度和相关性

        Args:
            entity_types: 新发现的实体类型列表
            entities: 所有实体列表

        Returns:
            添加了质量指标的实体类型列表
        """
        evaluated_types = []

        for entity_type in entity_types:
            type_name = entity_type['name']
            type_definition = entity_type['definition']
            examples = entity_type.get('examples', [])

            logger.info(f"评估实体类型 '{type_name}' 的质量...")

            # 格式化示例为可读文本
            example_text = []
            for i, example in enumerate(examples[:3]):  # 取前3个示例用于评估
                entity_id = example.get('id', '')
                entity_name = example.get('name', '')

                if not entity_name:
                    # 尝试从实体列表中找到该实体
                    entity_obj = next((e for e in entities if e.get('id') == entity_id), None)
                    if entity_obj:
                        entity_name = entity_obj.get('name', entity_id)

                example_text.append(f"Example {i+1}: {entity_name}")

            # 1. 置信度评估
            confidence_prompt = f"""
            Evaluate the factual confidence of this computer networks entity type:

            Entity type: "{type_name}"
            Definition: "{type_definition}"
            Examples:
            {chr(10).join(example_text)}

            I need you to assess how confident we can be that this entity type is technically accurate and useful for a computer networks knowledge graph.

            Consider factors like:
            1. Is this a well-established technical category in computer networking?
            2. Is there substantial evidence in computer networking literature supporting this type of entity?
            3. Would computer networking experts broadly agree with this entity category?
            4. Does it align with current technical understanding and provide educational value?

            Rate your confidence from 0.0 (completely uncertain) to 1.0 (extremely confident).
            Return only a single float value between 0.0 and 1.0.

            Example ratings:
            - 0.95-0.99: Well-established technical category with overwhelming evidence
            - 0.80-0.94: Strong technical consensus with substantial supporting evidence
            - 0.60-0.79: Reasonably supported entity category with some evidence base
            - 0.40-0.59: Mixed evidence or emerging concept
            - 0.20-0.39: Limited evidence, preliminary category
            - 0.01-0.19: Very weak evidence, highly speculative

            Return only the numeric value as a float between 0.0 and 1.0.
            """

            # 2. 清晰度评估
            clarity_prompt = f"""
            Evaluate the clarity and specificity of this computer networks entity type:

            Entity type: "{type_name}"
            Definition: "{type_definition}"
            Examples:
            {chr(10).join(example_text)}

            Assess how clear, specific, and unambiguous this entity type would be to computer networking experts and students.

            Consider these aspects:
            1. Is the entity type name precise and unambiguous?
            2. Is the definition specific and well-explained?
            3. Would networking experts interpret this entity type consistently?
            4. Are there any vague terms or imprecise language in the definition?
            5. Do the examples clearly illustrate the entity type?

            Rate clarity from 0.0 (very ambiguous) to 1.0 (perfectly clear).
            Return only a single float value between 0.0 and 1.0.

            Example ratings:
            - 0.95-0.99: Crystal clear type with highly specific definition and examples
            - 0.80-0.94: Very clear type with minor room for interpretation
            - 0.60-0.79: Mostly clear type with some potential ambiguity
            - 0.40-0.59: Type with moderate ambiguity or vagueness
            - 0.20-0.39: Significantly vague or ambiguous entity type
            - 0.01-0.19: Extremely vague type with minimal specificity

            Return only the numeric value as a float between 0.0 and 1.0.
            """

            # 3. 相关性评估
            relevance_prompt = f"""
            Evaluate how relevant this entity type is to the computer networks domain:

            Entity type: "{type_name}"
            Definition: "{type_definition}"
            Examples:
            {chr(10).join(example_text)}

            Assess how important and appropriate this entity type is for inclusion in a specialized computer networks knowledge graph.

            Consider these aspects:
            1. Is this entity type directly relevant to computer networks teaching and learning?
            2. Does this provide valuable categorization for computer networks concepts?
            3. Would this entity type be useful for computer networks education applications?
            4. Is this specialized knowledge that helps organize computer networks concepts?
            5. Does it align with how computer networks experts would categorize information?

            Rate relevance from 0.0 (completely irrelevant) to 1.0 (highly relevant).
            Return only a single float value between 0.0 and 1.0.

            Example ratings:
            - 0.95-0.99: Core computer networks categorization essential to understanding the field
            - 0.80-0.94: Highly relevant entity type for computer networks knowledge organization
            - 0.60-0.79: Moderately relevant entity type with clear applications
            - 0.40-0.59: Somewhat relevant but not central to the domain
            - 0.20-0.39: Marginally relevant entity type
            - 0.01-0.19: Largely irrelevant to the computer networks domain

            Return only the numeric value as a float between 0.0 and 1.0.
            """

            try:
                # 调用LLM获取评估结果 - 置信度
                confidence_response = self.llm_client.call(confidence_prompt, temperature=0.1, response_format="text")
                confidence_score = self._extract_float_score(confidence_response) or 0.5

                # 调用LLM获取评估结果 - 清晰度
                clarity_response = self.llm_client.call(clarity_prompt, temperature=0.1, response_format="text")
                clarity_score = self._extract_float_score(clarity_response) or 0.5

                # 调用LLM获取评估结果 - 相关性
                relevance_response = self.llm_client.call(relevance_prompt, temperature=0.1, response_format="text")
                relevance_score = self._extract_float_score(relevance_response) or 0.5

                # 添加评估指标到实体类型
                entity_type_with_metrics = entity_type.copy()
                entity_type_with_metrics["quality_metrics"] = {
                    "confidence": round(confidence_score, 2),
                    "clarity": round(clarity_score, 2),
                    "relevance": round(relevance_score, 2),
                    # 计算整合分数: confidence*0.4 + clarity*0.2 + relevance*0.4
                    "overall": round(confidence_score * 0.4 + clarity_score * 0.2 + relevance_score * 0.4, 2)
                }

                logger.info(f"实体类型 '{type_name}' 质量评估: 置信度={confidence_score:.2f}, 清晰度={clarity_score:.2f}, 相关性={relevance_score:.2f}")
                evaluated_types.append(entity_type_with_metrics)

            except Exception as e:
                logger.error(f"评估实体类型 '{type_name}' 时发生错误: {str(e)}")
                # 如果评估失败，仍然添加原始类型
                evaluated_types.append(entity_type)

        # 按整体质量分数排序，最高分在前
        evaluated_types.sort(key=lambda x: x.get("quality_metrics", {}).get("overall", 0), reverse=True)

        return evaluated_types

    def _evaluate_relationship_types_quality(self, relationship_types: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        评估新发现的关系类型的质量：置信度、清晰度和相关性

        Args:
            relationship_types: 新发现的关系类型列表
            entities: 所有实体列表

        Returns:
            添加了质量指标的关系类型列表
        """
        # 创建实体ID到实体的映射
        entity_map = {entity['id']: entity for entity in entities}

        evaluated_types = []

        for rel_type in relationship_types:
            type_name = rel_type['name']
            type_definition = rel_type['definition']
            examples = rel_type.get('examples', [])

            logger.info(f"评估关系类型 '{type_name}' 的质量...")

            # 格式化示例为可读文本
            example_text = []
            for i, example in enumerate(examples[:3]):  # 取前3个示例用于评估
                entity1 = example.get('entity1', '')
                entity2 = example.get('entity2', '')
                if not entity1 or not entity2:
                    source = example.get('source', '')
                    target = example.get('target', '')
                    if source and target:
                        entity1, entity2 = source, target

                # 找到实体名称
                entity1_name = next((entity['name'] for entity in entities if entity.get('id') == entity1), entity1)
                entity2_name = next((entity['name'] for entity in entities if entity.get('id') == entity2), entity2)

                example_text.append(f"Example {i+1}: {entity1_name} -> {type_name} -> {entity2_name}")

            # 1. 置信度评估
            confidence_prompt = f"""
            Evaluate the factual confidence of this computer networks relationship type:

            Relationship type: "{type_name}"
            Definition: "{type_definition}"
            Examples:
            {chr(10).join(example_text)}

            I need you to assess how confident we can be that this relationship type is technically accurate and useful for a computer networks knowledge graph.

            Consider factors like:
            1. Is this a well-established technical relationship category in computer networking?
            2. Is there substantial evidence in computer networking literature supporting this type of relationship?
            3. Would computer networking experts broadly agree with this relationship category?
            4. Does it align with current technical understanding and provide educational value?

            Rate your confidence from 0.0 (completely uncertain) to 1.0 (extremely confident).
            Return only a single float value between 0.0 and 1.0.

            Example ratings:
            - 0.95-0.99: Well-established technical relationship with overwhelming evidence
            - 0.80-0.94: Strong technical consensus with substantial supporting evidence
            - 0.60-0.79: Reasonably supported relationship category with some evidence base
            - 0.40-0.59: Mixed evidence or emerging concept
            - 0.20-0.39: Limited evidence, preliminary category
            - 0.01-0.19: Very weak evidence, highly speculative

            Return only the numeric value as a float between 0.0 and 1.0.
            """

            # 2. 清晰度评估
            clarity_prompt = f"""
            Evaluate the clarity and specificity of this computer networks relationship type:

            Relationship type: "{type_name}"
            Definition: "{type_definition}"
            Examples:
            {chr(10).join(example_text)}

            Assess how clear, specific, and unambiguous this relationship type would be to computer networking experts and students.

            Consider these aspects:
            1. Is the relationship type name precise and unambiguous?
            2. Is the definition specific and well-explained?
            3. Would networking experts interpret this relationship type consistently?
            4. Are there any vague terms or imprecise language in the definition?
            5. Do the examples clearly illustrate the relationship type?

            Rate clarity from 0.0 (very ambiguous) to 1.0 (perfectly clear).
            Return only a single float value between 0.0 and 1.0.

            Example ratings:
            - 0.95-0.99: Crystal clear type with highly specific definition and examples
            - 0.80-0.94: Very clear type with minor room for interpretation
            - 0.60-0.79: Mostly clear type with some potential ambiguity
            - 0.40-0.59: Type with moderate ambiguity or vagueness
            - 0.20-0.39: Significantly vague or ambiguous relationship type
            - 0.01-0.19: Extremely vague type with minimal specificity

            Return only the numeric value as a float between 0.0 and 1.0.
            """

            # 3. 相关性评估
            relevance_prompt = f"""
            Evaluate how relevant this relationship type is to the computer networks domain:

            Relationship type: "{type_name}"
            Definition: "{type_definition}"
            Examples:
            {chr(10).join(example_text)}

            Assess how important and appropriate this relationship type is for inclusion in a specialized computer networks knowledge graph.

            Consider these aspects:
            1. Is this relationship type directly relevant to computer networks teaching and learning?
            2. Does this provide valuable categorization for computer networks concepts?
            3. Would this relationship type be useful for computer networks education applications?
            4. Is this specialized knowledge that helps organize computer networks concepts?
            5. Does it align with how computer networks experts would categorize information?

            Rate relevance from 0.0 (completely irrelevant) to 1.0 (highly relevant).
            Return only a single float value between 0.0 and 1.0.

            Example ratings:
            - 0.95-0.99: Core computer networks relationship essential to understanding the field
            - 0.80-0.94: Highly relevant relationship type for computer networks knowledge organization
            - 0.60-0.79: Moderately relevant relationship type with clear applications
            - 0.40-0.59: Somewhat relevant but not central to the domain
            - 0.20-0.39: Marginally relevant relationship type
            - 0.01-0.19: Largely irrelevant to the computer networks domain

            Return only the numeric value as a float between 0.0 and 1.0.
            """

            try:
                # 调用LLM获取评估结果 - 置信度
                confidence_response = self.llm_client.call(confidence_prompt, temperature=0.1, response_format="text")
                confidence_score = self._extract_float_score(confidence_response) or 0.5

                # 调用LLM获取评估结果 - 清晰度
                clarity_response = self.llm_client.call(clarity_prompt, temperature=0.1, response_format="text")
                clarity_score = self._extract_float_score(clarity_response) or 0.5

                # 调用LLM获取评估结果 - 相关性
                relevance_response = self.llm_client.call(relevance_prompt, temperature=0.1, response_format="text")
                relevance_score = self._extract_float_score(relevance_response) or 0.5

                # 添加评估指标到关系类型
                rel_type_with_metrics = rel_type.copy()
                rel_type_with_metrics["quality_metrics"] = {
                    "confidence": round(confidence_score, 2),
                    "clarity": round(clarity_score, 2),
                    "relevance": round(relevance_score, 2),
                    # 计算整合分数: confidence*0.4 + clarity*0.2 + relevance*0.4
                    "overall": round(confidence_score * 0.4 + clarity_score * 0.2 + relevance_score * 0.4, 2)
                }

                logger.info(f"关系类型 '{type_name}' 质量评估: 置信度={confidence_score:.2f}, 清晰度={clarity_score:.2f}, 相关性={relevance_score:.2f}")
                evaluated_types.append(rel_type_with_metrics)

            except Exception as e:
                logger.error(f"评估关系类型 '{type_name}' 时发生错误: {str(e)}")
                # 如果评估失败，仍然添加原始类型
                evaluated_types.append(rel_type)

        # 按整体质量分数排序，最高分在前
        evaluated_types.sort(key=lambda x: x.get("quality_metrics", {}).get("overall", 0), reverse=True)

        return evaluated_types

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
        return None
