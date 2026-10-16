"""
Entity definer for defining entity types and relationships.
"""

import os
import json
import logging
import re
from typing import Dict, List, Any

from tqdm import tqdm

import config
from utils.llm_utils import LLMClient

logger = logging.getLogger(__name__)

class EntityDefiner:
    """
    Class for defining entity types and relationships.
    """

    def __init__(self, model_name: str = config.DEFAULT_MODEL, api_key: str = None, api_base: str = None):
        """
        Initialize the entity definer.

        Args:
            model_name: Name of the LLM model to use
            api_key: API key for the LLM service
            api_base: API base URL for the LLM service
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base

        # Initialize LLM client
        self.llm_client = LLMClient(model_name=model_name, api_key=api_key, api_base=api_base)

        logger.info(f"Initialized entity definer with model {model_name}")

        # Load definition prompt templates
        self.entity_definition_prompt = self._load_prompt_template("entity")
        self.relationship_definition_prompt = self._load_prompt_template("relationship")

    def define(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Define entity types and relationships.

        Args:
            extraction_result: Dictionary containing extracted entities and relationships

        Returns:
            Dictionary containing defined entities and relationships
        """
        logger.info(f"Defining entities and relationships from {extraction_result['filename']}")

        # 获取chapter信息，如果不存在则设置默认值
        chapter = ""
        # 尝试从第一个实体获取chapter信息
        if extraction_result.get("entities") and len(extraction_result["entities"]) > 0:
            chapter = extraction_result["entities"][0].get("chapter", "")

        # Create a clean result dictionary (only keep essential fields)
        result = {
            "filename": extraction_result["filename"],
            "chapter": chapter,  # 使用获取的chapter或默认值
            "entities": [],
            "relationships": []
        }

        # Define entities
        defined_entities = self.define_entities(extraction_result["entities"])
        result["entities"] = defined_entities

        # Define relationships
        defined_relationships = self.define_relationships(extraction_result["relationships"], defined_entities)
        result["relationships"] = defined_relationships

        return result

    def define_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Define entities.

        Args:
            entities: List of entities

        Returns:
            List of entities with definitions
        """
        logger.info(f"Defining {len(entities)} entities")

        defined_entities = []

        # Define each entity
        for entity in tqdm(entities, desc="Defining entities"):
            try:
                # Copy entity to avoid modifying the original
                defined_entity = entity.copy()

                # Skip entities that are already defined, but add significance if missing
                if "definition" in entity and entity["definition"]:
                    logger.info(f"Entity '{entity['name']}' already has a definition, checking for significance")

                    # If entity has definition but no significance, add significance
                    if "significance" not in entity:
                        significance = self._get_entity_significance(entity)
                        defined_entity["significance"] = significance
                        logger.info(f"Added significance {significance} for entity '{entity['name']}'")

                    defined_entities.append(defined_entity)
                    continue

                # Get entity definition and significance
                result = self._get_entity_definition_with_significance(entity)

                # Add definition and significance to entity
                defined_entity["definition"] = result["definition"]
                defined_entity["significance"] = result["significance"]

                defined_entities.append(defined_entity)
                logger.info(f"Defined entity '{entity['name']}': {result['definition'][:100]}... (Significance: {result['significance']})")

            except Exception as e:
                logger.error(f"Error defining entity '{entity['name']}': {e}")
                # Add entity without definition or with default significance
                if "definition" not in entity:
                    entity["definition"] = ""
                if "significance" not in entity:
                    entity["significance"] = 3  # Default medium importance
                defined_entities.append(entity)

        return defined_entities

    def define_relationships(self, relationships: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Define relationships between entities.

        Args:
            relationships: List of relationships
            entities: List of entities

        Returns:
            List of relationships with definitions
        """
        logger.info(f"Defining {len(relationships)} relationships")

        # Create entity ID to entity mapping
        entity_map = {entity["id"]: entity for entity in entities}

        defined_relationships = []

        # Define each relationship
        for relationship in tqdm(relationships, desc="Defining relationships"):
            try:
                # Skip relationships that are already defined
                if "definition" in relationship and relationship["definition"]:
                    logger.info(f"Relationship already has a definition, skipping")
                    defined_relationships.append(relationship)
                    continue

                # Get entity IDs
                entity1_id = relationship.get("entity1") or relationship.get("source")
                entity2_id = relationship.get("entity2") or relationship.get("target")

                # Get entities
                entity1 = entity_map.get(entity1_id)
                entity2 = entity_map.get(entity2_id)

                if entity1 and entity2:
                    # Get relationship definition
                    definition_data = self._get_relationship_definition(entity1, entity2, relationship["type"])

                    # Copy relationship and add definition
                    defined_relationship = relationship.copy()

                    # Set definition and strength from the response
                    defined_relationship["definition"] = definition_data.get("definition", "Relationship between these entities")

                    # 将原来的1-3的强度标准转换为0-1的标准
                    strength = definition_data.get("strength", 0.5)
                    defined_relationship["strength"] = strength

                    # Add suggested_type if available
                    if "suggested_type" in definition_data:
                        defined_relationship["suggested_type"] = definition_data["suggested_type"]
                        logger.info(f"Added suggested relationship type: {definition_data['suggested_type']}")

                    defined_relationships.append(defined_relationship)
                    logger.info(f"Defined relationship between '{entity1['name']}' and '{entity2['name']}': {defined_relationship['definition'][:100]}...")

                else:
                    logger.warning(f"Could not find entities for relationship: {relationship}")
                    # Add relationship without definition
                    defined_relationships.append(relationship)

            except Exception as e:
                logger.error(f"Error defining relationship: {e}")
                # Add relationship without definition
                defined_relationships.append(relationship)

        return defined_relationships

    def _extract_plain_text_definition(self, definition: str) -> str:
        """
        Extract plain text definition from response that might be in JSON format.

        Args:
            definition: Definition string that might be in JSON format

        Returns:
            Plain text definition
        """
        # Check if the definition is in JSON format
        if definition.strip().startswith('[') and definition.strip().endswith(']'):
            try:
                # Parse JSON
                data = json.loads(definition)

                # Extract definition field based on known formats
                if isinstance(data, list) and len(data) > 0:
                    item = data[0]

                    # Format for entity definitions
                    if "definition" in item:
                        return item["definition"]

                    # Format for relationship definitions
                    if "relationship_type" in item and "definition" in item:
                        return item["definition"]

                logger.warning(f"Unrecognized JSON definition format: {definition}")
                return definition  # Return original if format not recognized

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse definition as JSON: {definition}")
                return definition  # Return original if parsing fails

        # Return as is if not in JSON format
        return definition

    def _get_entity_definition_with_significance(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the definition and significance for an entity.

        Args:
            entity: Entity dictionary

        Returns:
            Dictionary with definition and significance
        """
        # 准备提示以定义实体并评估重要性
        entity_name = entity["name"]
        entity_type = entity["type"]
        entity_context = entity.get("context", "")  # 获取上下文，如果不存在则为空字符串

        # 使用更新的模板，纳入上下文信息和重要性评估
        prompt = self.entity_definition_prompt.format(
            entity_name=entity_name,
            entity_type=entity_type,
            context=entity_context
        )

        # 调用LLM API获取定义和重要性
        response = self.llm_client.call(prompt, response_format="json")

        try:
            # 解析JSON响应
            result = json.loads(response)

            # 提取定义和重要性
            definition = result.get("definition", "")
            significance = result.get("significance", 3)  # 默认为3（中等重要性）

            # 确保significance在1-5的范围内
            if not isinstance(significance, int) or significance < 1 or significance > 5:
                logger.warning(f"Invalid significance value: {significance}, defaulting to 3")
                significance = 3

            # 记录响应
            logger.info(f"LLM responded with entity definition and significance: {definition[:100]}... (Significance: {significance})")

            return {
                "definition": definition,
                "significance": significance
            }

        except json.JSONDecodeError:
            # 如果无法解析JSON，尝试提取纯文本定义并设置默认重要性
            logger.error(f"Failed to parse entity definition and significance: {response}")
            return {
                "definition": response.strip(),
                "significance": 3  # 默认为中等重要性
            }

    def _get_entity_significance(self, entity: Dict[str, Any]) -> int:
        """
        Get only the significance for an entity that already has a definition.

        Args:
            entity: Entity dictionary with definition

        Returns:
            Significance value (1-5)
        """
        # 准备提示以评估实体重要性
        entity_name = entity["name"]
        entity_type = entity["type"]
        entity_definition = entity.get("definition", "")

        prompt = f"""
        You are a computer network course instructor tasked with evaluating the importance of concepts in a computer networks course.

        Please evaluate the significance of the following entity for a computer networks course:

        Entity Name: {entity_name}
        Entity Type: {entity_type}
        Entity Definition: {entity_definition}

        Rate the significance of this entity to the computer networks field on a scale of 1-5:

        5 - Core concept: Fundamental to understanding computer networks, essential for all students (e.g., TCP/IP, routing protocols, OSI model)
        4 - Important concept: Necessary for comprehensive understanding of computer networks (e.g., subnetting, DNS, network security)
        3 - Relevant concept: Regularly encountered in computer networks, beneficial to understand (e.g., specific protocols, networking tools)
        2 - Peripheral concept: Related to computer networks but not central to the course (e.g., specific technologies or applications)
        1 - Marginal concept: Only tangentially related to computer networks (e.g., concepts from other fields occasionally mentioned)

        Return ONLY a single integer between 1 and 5 as your answer.
        """

        # 调用LLM API获取重要性
        response = self.llm_client.call(prompt, response_format="text")

        # 提取数字
        significance_match = re.search(r'[1-5]', response.strip())
        if significance_match:
            significance = int(significance_match.group(0))
        else:
            logger.warning(f"Could not extract significance value from response: {response}")
            significance = 3  # 默认为中等重要性

        return significance

    def _get_entity_definition(self, entity: Dict[str, Any]) -> str:
        """
        Get the definition for an entity. Now a wrapper for _get_entity_definition_with_significance.

        Args:
            entity: Entity dictionary

        Returns:
            Definition string
        """
        result = self._get_entity_definition_with_significance(entity)
        return result["definition"]

    def _get_relationship_definition(self, source_entity: Dict[str, Any], target_entity: Dict[str, Any], relationship_type: str) -> Dict[str, Any]:
        """
        Get the definition for a relationship between two entities.

        Args:
            source_entity: Source entity dictionary
            target_entity: Target entity dictionary
            relationship_type: Type of relationship

        Returns:
            Dictionary containing relationship definition
        """
        # 准备提示以定义关系
        source_name = source_entity["name"]
        target_name = target_entity["name"]
        source_type = source_entity["type"]
        target_type = target_entity["type"]
        source_context = source_entity.get("context", "")  # 获取源实体的上下文
        target_context = target_entity.get("context", "")  # 获取目标实体的上下文

        # 使用更新的模板，纳入上下文信息
        prompt = self.relationship_definition_prompt.format(
            source_name=source_name,
            target_name=target_name,
            relationship_type=relationship_type,
            source_type=source_type,
            target_type=target_type,
            source_context=source_context,
            target_context=target_context
        )

        # 调用LLM API获取定义
        response = self.llm_client.call(prompt)

        # 记录原始响应
        logger.info(f"LLM responded with relationship definition: {response[:200]}...")

        try:
            # 解析JSON响应
            data = json.loads(response)

            # 返回解析的数据
            result = {}
            if isinstance(data, dict):
                result = {
                    "definition": data.get("definition", "Relationship between these entities"),
                    "strength": data.get("strength", 0.5)
                }
                # Add suggested_type if it exists
                if "suggested_type" in data:
                    result["suggested_type"] = data["suggested_type"]
                return result
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                result = {
                    "definition": data[0].get("definition", "Relationship between these entities"),
                    "strength": data[0].get("strength", 0.5)
                }
                # Add suggested_type if it exists
                if "suggested_type" in data[0]:
                    result["suggested_type"] = data[0]["suggested_type"]
                return result
            else:
                logger.warning(f"Unexpected response format: {type(data)}")
                return {"definition": "Relationship between these entities", "strength": 0.5}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse relationship definition: {e}")
            logger.error(f"Response: {response}")

            # 返回默认定义
            return {"definition": "Relationship between these entities", "strength": 0.5}

    def _load_prompt_template(self, template_type: str) -> str:
        """
        Load a prompt template for entity or relationship definition.

        Args:
            template_type: Type of template ("entity" or "relationship")

        Returns:
            Template string
        """
        if template_type == "entity":
            return """
            You are a computer network course instructor tasked with defining concepts in a computer networks course and evaluating their importance.

            Please provide a concise definition of the entity AND rate its significance to computer networks:

            Entity Name: {entity_name}
            Entity Type: {entity_type}
            Context: {context}

            Guidelines for definition:
            1. Provide a clear, educational definition appropriate for computer network students.
            2. Use the context information to tailor your definition specifically to how this concept is used in the course.
            3. Focus exclusively on computer networking aspects of the concept.
            4. Keep your definition concise (2-4 sentences) but comprehensive.
            5. Include key characteristics, functions, or importance within computer networks.
            6. Avoid mentioning specific implementations unless they are standard.
            7. Consider the entity type when crafting your definition.
            8. If the entity is a technical term, include its purpose and how it works.

            Guidelines for significance rating:
            Rate the significance of this entity to the computer networks field on a scale of 1-5:

            5 - Core concept: Fundamental to understanding computer networks, essential for all students (e.g., TCP/IP, routing protocols, OSI model)
            4 - Important concept: Necessary for comprehensive understanding of computer networks (e.g., subnetting, DNS, network security)
            3 - Relevant concept: Regularly encountered in computer networks, beneficial to understand (e.g., specific protocols, networking tools)
            2 - Peripheral concept: Related to computer networks but not central to the course (e.g., specific technologies or applications)
            1 - Marginal concept: Only tangentially related to computer networks (e.g., concepts from other fields occasionally mentioned)

            Return your response in JSON format:
            {{
                "definition": "Your concise definition here",
                "significance": <integer between 1 and 5>
            }}
            """
        elif template_type == "relationship":
            return """
            You are a computer network course instructor tasked with defining relationships between networking concepts.

            Please analyze the relationship between these two entities:

            Source Entity: {source_name} (Type: {source_type})
            Source Context: {source_context}

            Target Entity: {target_name} (Type: {target_type})
            Target Context: {target_context}

            Relationship Type: {relationship_type}

            Define this relationship in the context of computer networking education. Consider:
            - How these concepts interact in networking systems
            - The hierarchical or functional relationship between them
            - How understanding one concept helps in understanding the other
            - The specific way the relationship type "{relationship_type}" applies
            - Use the provided context to make your definition more precise and relevant

            STANDARD RELATIONSHIP TYPES IN COMPUTER NETWORKS:
            1. is_a: Taxonomy relationship (e.g., "Ethernet is_a Network Technology", "Router is_a Network Device")
            2. part_of: Composition relationship (e.g., "TCP is part_of TCP/IP Suite", "Network Interface Card is part_of Computer Hardware")
            3. depends_on: Dependency relationship (e.g., "Web Browser depends_on HTTP", "Routing depends_on IP Addresses")
            4. equivalent_to: Equivalence relationship (e.g., "WWW is equivalent_to World Wide Web", "802.11 is equivalent_to WiFi")
            5. prerequisite_for: Educational prerequisite relationship (e.g., "OSI Model is prerequisite_for Protocol Analysis", "Binary Numbers are prerequisite_for IP Addressing")
            6. sibling_of: Parallel/peer relationship for related concepts at the same level (e.g., "TCP sibling_of UDP" as they are both Transport protocols, "Star Topology sibling_of Bus Topology" as they are both network topologies)
            7. related_to: General relationship type for concepts with a connection that doesn't fit the above types

            Your definition should align with one of these standard relationship types when possible, making it easier to categorize the relationship correctly.

            Return your response in JSON format:
            {{
                "definition": "Concise definition of the relationship (2-3 sentences)",
                "strength": <value between 0 and 1 indicating relationship strength>,
                "suggested_type": "one of the standard relationship types that best matches this relationship"
            }}

            Guidelines for strength value:
            - 0.9-1.0: Direct, fundamental relationship; one concept cannot exist without the other
            - 0.7-0.9: Strong conceptual or technical dependency
            - 0.5-0.7: Clear relationship but concepts can be understood separately
            - 0.3-0.5: Related but independent concepts
            - 0.1-0.3: Tangential or situational relationship

            Ensure your response is valid JSON.
            """
        else:
            return ""
