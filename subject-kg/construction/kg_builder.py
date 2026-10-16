"""
Knowledge graph builder for constructing the final knowledge graph.
"""

import os
import json
import logging
from typing import Dict, List, Any, Set

import networkx as nx
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """
    Class for building the knowledge graph.
    """

    def __init__(self):
        """
        Initialize the knowledge graph builder.
        """
        logger.info("Initialized knowledge graph builder")

    def build(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build the knowledge graph.

        Args:
            entities: List of entities
            relationships: List of relationships

        Returns:
            Dictionary containing the knowledge graph
        """
        logger.info("Building knowledge graph")

        # Filter out unverified entities and relationships
        verified_entities = [entity for entity in entities if entity.get("verified", True)]
        verified_relationships = [rel for rel in relationships if rel.get("verified", True)]

        logger.info(f"Building knowledge graph with {len(verified_entities)} entities and {len(verified_relationships)} relationships")

        # Calculate relationship counts for each entity
        entity_relationship_counts = self._calculate_relationship_counts(verified_relationships)

        # Log relationship counts for debugging
        entity_names = {entity["id"]: entity["name"] for entity in verified_entities}
        for entity_id, count in entity_relationship_counts.items():
            entity_name = entity_names.get(entity_id, entity_id)
            logger.info(f"Entity {entity_name} (ID: {entity_id}) has {count} relationships")

        # Clean entity data - only keep necessary fields
        clean_entities = []
        for entity in verified_entities:
            clean_entity = {
                "id": entity["id"],
                "name": entity["name"],
                "type": entity.get("explored_type", entity.get("canonical_type", entity["type"])),
                "definition": entity.get("definition", ""),
                "chapter": entity.get("chapter", ""),
                "page_number": entity.get("first_occurrence_page", entity.get("page_number", 0)),
                "relationship_count": entity_relationship_counts.get(entity["id"], 0),  # Add relationship count
                "first_occurrence": entity.get("first_occurrence", True),  # Preserve first_occurrence flag
                "significance": entity.get("significance", 3)  # Add significance attribute with default value 3
            }

            # Only add optional fields if they exist and are needed
            if "occurrences" in entity:
                clean_entity["occurrences"] = entity["occurrences"]

            clean_entities.append(clean_entity)

            # Log entity relationship count for debugging
            logger.info(f"Added entity {entity['name']} with relationship_count: {clean_entity['relationship_count']} and significance: {clean_entity['significance']}")

        # Clean relationship data - only keep necessary fields
        clean_relationships = []
        for rel in verified_relationships:
            clean_rel = {
                "id": rel["id"],
                "source": rel["source"],
                "target": rel["target"],
                "chapter": rel.get("chapter", ""),
                "page_number": rel.get("page_number", 0)
            }

            # 获取关系类型，优先使用规范化的类型，如果都不存在则使用默认related_to
            rel_type = rel.get("explored_type", rel.get("canonical_type", rel.get("type", "related_to")))
            clean_rel["type"] = rel_type

            # 保留strength属性，如果不存在则默认为1
            clean_rel["strength"] = rel.get("strength", 1)

            # 添加定义，如果存在
            if "definition" in rel:
                clean_rel["definition"] = rel["definition"]

            # Add source and target names if available
            if "source_name" in rel:
                clean_rel["source_name"] = rel["source_name"]

            if "target_name" in rel:
                clean_rel["target_name"] = rel["target_name"]

            clean_relationships.append(clean_rel)

        # Create a graph
        graph = self._create_graph(clean_entities, clean_relationships)

        # Analyze graph
        graph_analysis = self._analyze_graph(graph)

        # Create the knowledge graph
        knowledge_graph = {
            "entities": clean_entities,
            "relationships": clean_relationships,
            "analysis": graph_analysis
        }

        return knowledge_graph

    def _calculate_relationship_counts(self, relationships: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Calculate the number of relationships for each entity.

        Args:
            relationships: List of relationships

        Returns:
            Dictionary mapping entity IDs to their relationship counts
        """
        entity_counts = {}

        # 增加详细日志记录
        logger.info(f"Calculating relationship counts for {len(relationships)} relationships")

        for rel in relationships:
            source_id = rel["source"]
            target_id = rel["target"]

            # 记录处理的关系
            logger.debug(f"Processing relationship: {rel.get('source_name', source_id)} -> {rel.get('target_name', target_id)}")

            # Increment the count for source entity
            entity_counts[source_id] = entity_counts.get(source_id, 0) + 1

            # Increment the count for target entity
            entity_counts[target_id] = entity_counts.get(target_id, 0) + 1

        # 记录实体的关系计数
        for entity_id, count in entity_counts.items():
            logger.debug(f"Entity {entity_id} has {count} relationships")

        return entity_counts

    def _create_graph(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Create a NetworkX graph from entities and relationships.

        Args:
            entities: List of entities
            relationships: List of relationships

        Returns:
            NetworkX directed graph
        """
        # Create a directed graph
        graph = nx.DiGraph()

        # Create entity lookup
        entity_map = {entity["id"]: entity for entity in entities}

        # Add nodes (entities)
        for entity in entities:
            graph.add_node(
                entity["id"],
                name=entity["name"],
                type=entity.get("explored_type", entity.get("canonical_type", entity["type"])),
                definition=entity.get("definition", ""),
                chapter=entity.get("chapter", ""),
                first_occurrence_page=entity.get("first_occurrence_page", entity.get("page_number", 0)),
                relationship_count=entity.get("relationship_count", 0),  # Add relationship count to node
                significance=entity.get("significance", 3)  # Add significance to node
            )

        # Add edges (relationships)
        for relationship in relationships:
            source_id = relationship["source"]
            target_id = relationship["target"]

            # Check if both source and target entities exist
            if source_id in entity_map and target_id in entity_map:
                # 获取关系类型，优先使用规范化的类型，如果都不存在则使用默认related_to
                rel_type = relationship.get("explored_type", relationship.get("canonical_type", relationship.get("type", "related_to")))

                edge_attrs = {
                    "type": rel_type,
                    "chapter": relationship.get("chapter", ""),
                    "page_number": relationship.get("page_number", 0),
                    "strength": relationship.get("strength", 1)  # 添加关系强度
                }

                # 添加定义，如果存在
                if "definition" in relationship:
                    edge_attrs["definition"] = relationship["definition"]

                graph.add_edge(
                    source_id,
                    target_id,
                    **edge_attrs
                )

        return graph

    def _analyze_graph(self, G) -> Dict[str, Any]:
        """
        Analyze the graph to extract statistics.

        Args:
            G: NetworkX graph

        Returns:
            Dictionary of graph statistics
        """
        # Count number of entities and relationships
        entity_count = len(G.nodes())
        relationship_count = len(G.edges())

        # Count number of connected components
        connected_components = list(nx.weakly_connected_components(G))
        num_connected_components = len(connected_components)

        # Calculate average degree
        if entity_count > 0:
            total_degree = sum(dict(G.degree()).values())
            avg_degree = total_degree / entity_count
        else:
            avg_degree = 0

        # Count isolated entities (degree 0)
        isolated_entities = [n for n, d in G.degree() if d == 0]
        num_isolated_entities = len(isolated_entities)

        # Calculate entity type distribution
        entity_type_dist = {}
        for node, data in G.nodes(data=True):
            entity_type = data.get("canonical_type", data.get("type", "unknown"))
            entity_type_dist[entity_type] = entity_type_dist.get(entity_type, 0) + 1

        # Calculate relationship type distribution
        relationship_type_dist = {}
        for u, v, data in G.edges(data=True):
            # 获取关系类型，优先使用规范化的类型
            relationship_type = data.get("type", "related_to")
            relationship_type_dist[relationship_type] = relationship_type_dist.get(relationship_type, 0) + 1

        # 计算关系强度分布
        relationship_strength_dist = {1: 0, 2: 0, 3: 0}
        for u, v, data in G.edges(data=True):
            strength = data.get("strength", 1)
            relationship_strength_dist[strength] = relationship_strength_dist.get(strength, 0) + 1

        # 计算平均关系强度
        total_strength = sum(data.get("strength", 1) for u, v, data in G.edges(data=True))
        avg_strength = total_strength / relationship_count if relationship_count > 0 else 0

        # Graph density (proportion of actual edges to potential edges)
        if entity_count > 1:
            density = relationship_count / (entity_count * (entity_count - 1))
        else:
            density = 0

        # Return all statistics in a dictionary
        return {
            "entity_count": entity_count,
            "relationship_count": relationship_count,
            "num_connected_components": num_connected_components,
            "avg_degree": avg_degree,
            "num_isolated_entities": num_isolated_entities,
            "entity_type_distribution": entity_type_dist,
            "relationship_type_distribution": relationship_type_dist,
            "relationship_strength_distribution": relationship_strength_dist,
            "avg_relationship_strength": avg_strength,
            "density": density
        }
