"""
Relationship canonicalizer for standardizing relationships.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from tqdm import tqdm
import re
import datetime

# Add project root to path so we can import modules
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.llm_utils import LLMClient
import config
from config import RELATIONSHIP_TYPES

logger = logging.getLogger(__name__)

class RelationshipCanonicalizer:
    """
    Class for canonicalizing relationships.
    """

    def __init__(self, model_name: str = config.DEFAULT_MODEL, api_key: str = None, api_base: str = None):
        """
        Initialize the relationship canonicalizer.

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

        logger.info(f"Initialized relationship canonicalizer with model {model_name}")

        # Load canonicalization prompt templates
        self.relationship_canonicalization_prompt = self._load_prompt_template("relationship_canonicalization")

        # Define standard relationship types
        self.standard_relationship_types = config.RELATIONSHIP_TYPES

    def canonicalize(self, definition_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Canonicalize relationships, preserving entity types as-is.

        Args:
            definition_result: Dictionary containing defined entities and relationships

        Returns:
            Dictionary containing canonicalized relationships with original entities
        """
        logger.info(f"Canonicalizing relationships from {definition_result.get('filename', 'unknown')}")

        # 检查输入的关系中有多少含有定义
        relationships = definition_result.get("relationships", [])
        entities = definition_result.get("entities", [])

        relationships_with_def = sum(1 for rel in relationships if rel.get("definition", ""))
        logger.info(f"输入的关系总数: {len(relationships)}, 其中有定义的关系数: {relationships_with_def}")

        # 记录几个有定义的关系示例
        sample_size = min(3, relationships_with_def)
        if sample_size > 0:
            sample_rels = [rel for rel in relationships if rel.get("definition", "")][:sample_size]
            for i, rel in enumerate(sample_rels):
                logger.info(f"关系示例 {i+1}: {rel.get('source_name', '未知')} -> {rel.get('target_name', '未知')}")
                logger.info(f"定义: {rel.get('definition', '')[:100]}...")

        result = definition_result.copy()

        # 保留原始实体，不进行规范化
        result["entities"] = entities

        # 只规范化关系
        canonicalized_relationships = self._canonicalize_relationships(relationships, entities)
        result["relationships"] = canonicalized_relationships

        # 检查输出的关系中有多少含有定义
        output_with_def = sum(1 for rel in canonicalized_relationships if rel.get("definition", ""))
        logger.info(f"输出的关系总数: {len(canonicalized_relationships)}, 其中有定义的关系数: {output_with_def}")

        # 如果输出缺少定义，但输入有定义，可能是处理过程丢失了定义信息
        if relationships_with_def > 0 and output_with_def < relationships_with_def:
            logger.warning(f"处理过程中丢失了 {relationships_with_def - output_with_def} 个关系定义")

            # 尝试恢复丢失的定义
            relationship_dict = {f"{rel.get('source', '')}:{rel.get('target', '')}": rel for rel in relationships}

            for rel in canonicalized_relationships:
                key = f"{rel.get('source', '')}:{rel.get('target', '')}"
                if key in relationship_dict and not rel.get("definition") and relationship_dict[key].get("definition"):
                    rel["definition"] = relationship_dict[key]["definition"]
                    logger.info(f"恢复了关系 {rel.get('source_name', '')} -> {rel.get('target_name', '')} 的定义")

            # 再次检查
            fixed_with_def = sum(1 for rel in canonicalized_relationships if rel.get("definition", ""))
            logger.info(f"修复后，有定义的关系数: {fixed_with_def}")

        # 保存规范化结果到文件
        self._save_canonicalization_output(result)

        return result

    def _save_canonicalization_output(self, result: Dict[str, Any]) -> None:
        """
        保存规范化结果到输出目录。

        Args:
            result: 包含规范化后实体和关系的字典
        """
        # 确保输出目录存在
        output_dir = config.CANONICALIZATION_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        # 构建输出文件路径
        filename = result.get("filename", "combined_data")
        output_path = os.path.join(output_dir, f"{filename}_canonicalized.json")

        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"规范化结果已保存到 {output_path}")

        # 保存一个单独的combined文件用于后续处理
        combined_path = os.path.join(output_dir, "combined_canonicalized.json")
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"合并的规范化结果已保存到 {combined_path}")

    def _canonicalize_relationships(self, relationships: List[Dict[str, Any]],
                                   entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Canonicalize relationships, determine relationship types and directions.

        Args:
            relationships: List of undirected entity pair relationships
            entities: List of entities

        Returns:
            List of canonicalized relationships with direction (source -> target)
        """
        # Create entity lookup dictionary
        entity_dict = {entity["id"]: entity for entity in entities}

        canonicalized_relationships = []

        for relationship in tqdm(relationships, desc="Canonicalizing relationships"):
            canonicalized_relationship = relationship.copy()

            # Get entity pair
            entity1 = entity_dict.get(relationship["entity1"])
            entity2 = entity_dict.get(relationship["entity2"])

            if entity1 and entity2:
                # Get relationship definition
                relationship_definition = relationship.get("definition", "")
                if relationship_definition:
                    logger.info(f"Relationship definition length: {len(relationship_definition)} characters")
                    if len(relationship_definition) > 100:
                        logger.info(f"Definition snippet: {relationship_definition[:100]}...")
                else:
                    logger.warning(f"Relationship between {entity1['name']} and {entity2['name']} has no definition")

                # Ensure the canonicalized relationship preserves the definition
                canonicalized_relationship["definition"] = relationship_definition

                # Check if there's a suggested_type from the definer module
                suggested_type = relationship.get("suggested_type")
                if suggested_type and suggested_type in self.standard_relationship_types:
                    logger.info(f"Using suggested relationship type from definer: {suggested_type}")

                # Determine relationship type and direction using LLM
                relationship_info = self._determine_relationship_type_and_direction(
                    entity1["name"],
                    entity2["name"],
                    relationship_definition
                )

                # Set the relationship type and direction
                canonical_type = relationship_info["type"]
                direction = relationship_info["direction"]

                # If we have a suggested_type and the LLM returned "related_to", prefer the suggested_type
                if suggested_type and canonical_type == "related_to" and suggested_type != "related_to":
                    logger.info(f"Using suggested type '{suggested_type}' instead of 'related_to'")
                    canonical_type = suggested_type

                # For symmetric relationship types (equivalent_to, sibling_of), direction doesn't matter
                if canonical_type in ["equivalent_to", "sibling_of"]:
                    logger.info(f"Detected symmetric relationship type: {canonical_type}")
                    # Keep current direction, it doesn't matter for these types

                logger.info(f"Determined relationship: type={canonical_type}, direction={direction}")

                # Apply the determined direction
                if direction == "entity1_to_entity2":
                    canonicalized_relationship["source"] = relationship["entity1"]
                    canonicalized_relationship["target"] = relationship["entity2"]
                    canonicalized_relationship["source_name"] = entity1["name"]
                    canonicalized_relationship["target_name"] = entity2["name"]
                else:  # entity2_to_entity1
                    canonicalized_relationship["source"] = relationship["entity2"]
                    canonicalized_relationship["target"] = relationship["entity1"]
                    canonicalized_relationship["source_name"] = entity2["name"]
                    canonicalized_relationship["target_name"] = entity1["name"]

                # Set the relationship type
                canonicalized_relationship["type"] = canonical_type

                # Clean up entity pair fields that are no longer needed
                if "entity1" in canonicalized_relationship:
                    del canonicalized_relationship["entity1"]
                if "entity2" in canonicalized_relationship:
                    del canonicalized_relationship["entity2"]
                if "entity1_name" in canonicalized_relationship:
                    del canonicalized_relationship["entity1_name"]
                if "entity2_name" in canonicalized_relationship:
                    del canonicalized_relationship["entity2_name"]

                # Ensure source and target fields exist for compatibility
                if "source" not in canonicalized_relationship:
                    canonicalized_relationship["source"] = canonicalized_relationship.get("entity1")
                if "target" not in canonicalized_relationship:
                    canonicalized_relationship["target"] = canonicalized_relationship.get("entity2")
                if "source_name" not in canonicalized_relationship:
                    canonicalized_relationship["source_name"] = entity1["name"]
                if "target_name" not in canonicalized_relationship:
                    canonicalized_relationship["target_name"] = entity2["name"]

                canonicalized_relationships.append(canonicalized_relationship)

        # Count relationship types
        type_counts = {}
        for rel in canonicalized_relationships:
            rel_type = rel.get("type", "unknown")
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1

        logger.info(f"Relationship type distribution after canonicalization: {type_counts}")

        # Count relationships with definitions
        relations_with_definition = sum(1 for rel in canonicalized_relationships if rel.get("definition", ""))
        logger.info(f"Relationships with definitions: {relations_with_definition}/{len(canonicalized_relationships)}")

        return canonicalized_relationships

    def _determine_relationship_type_and_direction(self, entity1_name: str, entity2_name: str, relationship_definition: str) -> Dict[str, str]:
        """
        Determine relationship type and direction using LLM analysis.

        Args:
            entity1_name: Name of entity 1
            entity2_name: Name of entity 2
            relationship_definition: Relationship definition

        Returns:
            Dictionary with relationship type and direction
        """
        logger.info(f"Analyzing relationship between {entity1_name} and {entity2_name}")

        # Check if the relationship definition has a suggested_type
        # Try to extract suggested_type from the JSON definition if it's embedded
        suggested_type = None
        if relationship_definition:
            try:
                if "{" in relationship_definition and "}" in relationship_definition:
                    # Try to find and parse JSON in the definition
                    start = relationship_definition.find('{')
                    end = relationship_definition.rfind('}') + 1
                    if start >= 0 and end > start:
                        json_str = relationship_definition[start:end]
                        data = json.loads(json_str)
                        if isinstance(data, dict) and "suggested_type" in data:
                            suggested_type = data["suggested_type"]
                            logger.info(f"Found suggested_type in definition: {suggested_type}")
            except json.JSONDecodeError:
                pass  # Ignore parsing errors

        # If there's a suggested_type and it's in our standard types, use it
        if suggested_type and suggested_type in self.standard_relationship_types:
            logger.info(f"Using suggested relationship type: {suggested_type}")

            # For sibling_of relationships, direction isn't important
            if suggested_type == "sibling_of" or suggested_type == "equivalent_to":
                return {
                    "type": suggested_type,
                    "direction": "entity1_to_entity2"  # Default direction for symmetric relationships
                }

            # For other relationship types, still need to determine direction
            # Continue with normal processing, but pass the suggested type to help

        # Use extended prompt that asks for both type and direction
        prompt = self._load_prompt_template("relationship_type_and_direction").format(
            entity1_name=entity1_name,
            entity2_name=entity2_name,
            relationship_definition=relationship_definition if relationship_definition else "No definition available."
        )

        # Call LLM
        try:
            response = self.llm_client.call(prompt)
            logger.info(f"LLM response: {response}")

            # Parse the response to get type and direction
            result = self._parse_type_and_direction_response(response, entity1_name, entity2_name)

            # If direction is unclear, and relationship type is complex (not is_a or part_of),
            # default to a more general relationship type
            if suggested_type != "related_to" and result["type"] != "related_to":
                # Check if it's hard to determine direction based on the response
                if ("unclear" in response.lower() or "ambiguous" in response.lower() or
                    "difficult to determine" in response.lower()):
                    logger.info("Direction is unclear, defaulting to 'related_to' relationship type")
                    result["type"] = "related_to"

            # If we have a suggested_type that's more specific than "related_to", consider using it
            if suggested_type and suggested_type != "related_to" and result["type"] == "related_to":
                logger.info(f"Using suggested_type '{suggested_type}' instead of 'related_to'")
                result["type"] = suggested_type

            return result

        except Exception as e:
            logger.error(f"Error determining relationship type and direction: {str(e)}")
            # Return default values on error

            # If we have a suggested_type, use it
            if suggested_type:
                return {
                    "type": suggested_type,
                    "direction": "entity1_to_entity2"  # Default direction
                }

            return {
                "type": "related_to",
                "direction": "entity1_to_entity2"  # Default direction
            }

    def _parse_type_and_direction_response(self, response: str, entity1_name: str, entity2_name: str) -> Dict[str, str]:
        """
        Parse LLM response to extract relationship type and direction.

        Args:
            response: LLM response
            entity1_name: Name of entity 1
            entity2_name: Name of entity 2

        Returns:
            Dictionary with relationship type and direction
        """
        # Clean up response
        clean_response = response.strip()

        # Try to parse as JSON
        try:
            data = json.loads(clean_response)
            if isinstance(data, dict):
                rel_type = data.get("type", "related_to")
                direction = data.get("direction", "entity1_to_entity2")

                # Validate the relationship type
                if rel_type not in self.standard_relationship_types:
                    rel_type = self._parse_canonical_type(rel_type)

                # Validate the direction
                if direction not in ["entity1_to_entity2", "entity2_to_entity1"]:
                    logger.warning(f"Invalid direction: {direction}. Using default: entity1_to_entity2")
                    direction = "entity1_to_entity2"

                return {
                    "type": rel_type,
                    "direction": direction
                }
        except json.JSONDecodeError:
            # If not JSON, try to extract from text
            pass

        # Default to parsing just the type and use entity1_to_entity2 direction
        rel_type = self._parse_canonical_type(clean_response)

        # Look for direction indicators in the text
        direction = "entity1_to_entity2"  # Default direction

        # Check for direction indicators
        entity1_lower = entity1_name.lower()
        entity2_lower = entity2_name.lower()

        # Look for patterns indicating direction
        for direction_pattern in [
            f"{entity1_lower} is part of {entity2_lower}",
            f"{entity1_lower} depends on {entity2_lower}",
            f"{entity1_lower} is a {entity2_lower}",
            f"{entity1_lower} is an example of {entity2_lower}",
            f"{entity1_lower} belongs to {entity2_lower}"
        ]:
            if direction_pattern in clean_response.lower():
                direction = "entity1_to_entity2"
                break

        for reverse_pattern in [
            f"{entity2_lower} is part of {entity1_lower}",
            f"{entity2_lower} depends on {entity1_lower}",
            f"{entity2_lower} is a {entity1_lower}",
            f"{entity2_lower} is an example of {entity1_lower}",
            f"{entity2_lower} belongs to {entity1_lower}"
        ]:
            if reverse_pattern in clean_response.lower():
                direction = "entity2_to_entity1"
                break

        return {
            "type": rel_type,
            "direction": direction
        }

    def _load_prompt_template(self, template_name: str) -> str:
        """
        Load a prompt template.

        Args:
            template_name: Name of the template

        Returns:
            Prompt template string
        """
        # In a real implementation, you would load templates from files

        if template_name == "relationship_canonicalization":
            return """
            TASK: Analyze the relationship between computer network entities and determine the most specific and appropriate relationship type.

            Source Entity: {source_name}
            Target Entity: {target_name}
            Relationship Definition: {relationship_definition}

            AVAILABLE RELATIONSHIP TYPES:

            1. part_of
            - Definition: When one entity is a component, element, or subset of another entity
            - Examples:
              * Network Interface Card is part_of Computer Hardware
              * TCP is part_of TCP/IP Suite
              * Subnet is part_of Network

            2. is_a
            - Definition: When one entity is a type, instance, category, or example of another entity
            - Examples:
              * Ethernet is_a Network Technology
              * Router is_a Network Device
              * FTP is_a Application Protocol

            3. equivalent_to
            - Definition: When entities represent the same concept or are functionally identical
            - Examples:
              * WWW is equivalent_to World Wide Web
              * 802.11 is equivalent_to WiFi
              * URL is equivalent_to Uniform Resource Locator

            4. depends_on
            - Definition: When one entity requires, relies on, or utilizes another entity to function
            - Examples:
              * Web Browser depends_on HTTP
              * Routing depends_on IP Addresses
              * Network Security depends_on Encryption

            5. prerequisite_for
            - Definition: When understanding one entity is required before learning another (educational sequence)
            - Examples:
              * Network Topology is prerequisite_for Network Design
              * Binary Numbers are prerequisite_for IP Addressing
              * OSI Model is prerequisite_for Protocol Analysis

            6. sibling_of
            - Definition: When entities are related concepts at the same hierarchical level or are different examples of the same parent concept
            - Examples:
              * TCP sibling_of UDP (both are Transport protocols)
              * Star Topology sibling_of Bus Topology (both are network topologies)
              * IPv4 sibling_of IPv6 (both are Internet Protocol versions)

            7. related_to
            - Definition: When entities have a general connection but none of the above specific relationships apply
            - Examples:
              * Bandwidth is related_to Network Performance
              * Firewalls are related_to Network Security

            IMPORTANT: Choose the MOST SPECIFIC relationship type that accurately describes how these entities are related.
            Only use "related_to" if none of the more specific relationships apply.

            OUTPUT: Respond with ONLY the name of the relationship type (e.g., "part_of"). Do not include any explanation or other text.
            """
        elif template_name == "relationship_type_and_direction":
            return """
            TASK: Analyze the relationship between two computer network entities to determine both the relationship TYPE and DIRECTION.

            Entity 1: {entity1_name}
            Entity 2: {entity2_name}
            Relationship Definition: {relationship_definition}

            STEP 1: Determine the most specific and appropriate RELATIONSHIP TYPE.

            AVAILABLE RELATIONSHIP TYPES:

            1. part_of
            - Definition: When one entity is a component, element, or subset of another entity
            - Examples:
              * Network Interface Card is part_of Computer Hardware
              * TCP is part_of TCP/IP Suite
              * Subnet is part_of Network

            2. is_a
            - Definition: When one entity is a type, instance, category, or example of another entity
            - Examples:
              * Ethernet is_a Network Technology
              * Router is_a Network Device
              * FTP is_a Application Protocol

            3. equivalent_to
            - Definition: When entities represent the same concept or are functionally identical
            - Examples:
              * WWW is equivalent_to World Wide Web
              * 802.11 is equivalent_to WiFi
              * URL is equivalent_to Uniform Resource Locator

            4. depends_on
            - Definition: When one entity requires, relies on, or utilizes another entity to function
            - Examples:
              * Web Browser depends_on HTTP
              * Routing depends_on IP Addresses
              * Network Security depends_on Encryption

            5. prerequisite_for
            - Definition: When understanding one entity is required before learning another (educational sequence)
            - Examples:
              * Network Topology is prerequisite_for Network Design
              * Binary Numbers are prerequisite_for IP Addressing
              * OSI Model is prerequisite_for Protocol Analysis

            6. sibling_of
            - Definition: When entities are related concepts at the same hierarchical level or are different examples of the same parent concept
            - Examples:
              * TCP sibling_of UDP (both are Transport protocols)
              * Star Topology sibling_of Bus Topology (both are network topologies)
              * IPv4 sibling_of IPv6 (both are Internet Protocol versions)

            7. related_to
            - Definition: When entities have a general connection but none of the above specific relationships apply
            - Examples:
              * Bandwidth is related_to Network Performance
              * Firewalls are related_to Network Security

            STEP 2: Determine the DIRECTION of the relationship:

            For the direction, consider:
            - For part_of: Which is the component and which is the whole? The component is the source.
            - For is_a: Which is the specific instance and which is the general category? The specific is the source.
            - For depends_on: Which entity relies on the other? The dependent entity is the source.
            - For prerequisite_for: Which must be learned first? The prerequisite is the source.
            - For equivalent_to: Direction doesn't matter strongly, but choose based on which is more commonly used.
            - For sibling_of: Direction doesn't matter, as this is a symmetrical relationship.
            - For related_to: Choose the more specific entity as the source when possible.

            If you find it difficult to determine the direction with confidence, use the relationship type "related_to" instead.

            OUTPUT FORMAT:
            Respond with a JSON object containing two properties:
            1. "type": One of the seven relationship types above
            2. "direction": Either "entity1_to_entity2" or "entity2_to_entity1"

            For example:
            {{
                "type": "part_of",
                "direction": "entity1_to_entity2"
            }}

            This would mean "Entity 1 is part_of Entity 2"

            IMPORTANT: Choose the MOST SPECIFIC relationship type and the CORRECT direction based on the definitions above.
            If direction is unclear, prefer using "related_to" as the relationship type.
            """
        else:
            return ""

    def _get_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using simple string matching.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score
        """
        # 使用简单的Jaccard相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _parse_canonical_type(self, response: str) -> str:
        """
        Parse response to determine the canonical relationship type.

        Args:
            response: LLM response text

        Returns:
            Canonical relationship type
        """
        # Remove whitespace and convert to lowercase
        clean_response = response.strip().lower()

        # Check for standard relationship types directly
        for rel_type in self.standard_relationship_types:
            if rel_type.lower() in clean_response:
                return rel_type

        # Check for common patterns that map to standard types
        mappings = {
            "part": "part_of",
            "component": "part_of",
            "subset": "part_of",
            "element": "part_of",

            "is a": "is_a",
            "type of": "is_a",
            "kind of": "is_a",
            "instance of": "is_a",
            "category of": "is_a",

            "equivalent": "equivalent_to",
            "same as": "equivalent_to",
            "identical": "equivalent_to",
            "equal": "equivalent_to",

            "depends": "depends_on",
            "require": "depends_on",
            "need": "depends_on",
            "rely": "depends_on",
            "utilize": "depends_on",

            "prerequisite": "prerequisite_for",
            "before": "prerequisite_for",
            "prior": "prerequisite_for",
            "foundation": "prerequisite_for",

            "sibling": "sibling_of",
            "peer": "sibling_of",
            "parallel": "sibling_of",
            "similar concept": "sibling_of",
            "at the same level": "sibling_of",
            "both are": "sibling_of",

            "relate": "related_to",
            "connect": "related_to",
            "associate": "related_to",
            "link": "related_to"
        }

        for key, value in mappings.items():
            if key in clean_response:
                return value

        # Default to general relationship type
        logger.warning(f"Unable to parse relationship type, using default 'related_to'. Original response: {response}")
        return "related_to"
