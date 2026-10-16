"""
Entity extractor for identifying entities and relationships from preprocessed text.
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Tuple

from tqdm import tqdm

import config
from utils.llm_utils import LLMClient
from preprocessing.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    Class for extracting entities (knowledge points) and relationships from preprocessed text.
    """

    def __init__(self, model_name: str = config.DEFAULT_MODEL, api_key: str = None, api_base: str = None):
        """
        Initialize the entity extractor.

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

        logger.info(f"Initialized entity extractor with model {model_name}")

        # Load extraction prompt templates
        self.entity_prompt = self._load_prompt_template("entity_extraction")
        self.relationship_prompt = self._load_prompt_template("relationship_extraction")

    def extract(self, preprocessed_data: Dict[str, Any], file_index: int = 0) -> Dict[str, Any]:
        """
        Extract entities and relationships from preprocessed data.

        Args:
            preprocessed_data: Dictionary containing preprocessed data
            file_index: 文件索引号，用于创建全局唯一的实体ID

        Returns:
            Dictionary containing extracted entities and relationships
        """
        # 记录当前文件名，如果可用
        self.current_filename = preprocessed_data.get("filename", "unknown")
        logger.info(f"Extracting entities and relationships from {self.current_filename} (file_index: {file_index})")

        # 保存文件索引，用于创建唯一ID
        self.file_index = file_index

        # Initialize output structure
        output = {
            "filename": self.current_filename,
            "chapter": preprocessed_data.get("chapter", "Unknown"),
            "entities": [],
            "relationships": []
        }

        # 第一步：对所有页面抽取实体（不抽取关系）
        all_entities = []
        all_page_entities = {}  # 页码到实体列表的映射，用于后续按标题分组抽取关系

        if "pages" in preprocessed_data:
            pages = preprocessed_data["pages"]
            chapter = preprocessed_data.get("chapter", "Unknown")

            # 遍历每一页抽取实体
            for page in pages:
                page_number = page.get("page_number", 0)
                # 存储页面标题以用于上下文
                self.current_title = page.get("title", "")

                # 只抽取实体，暂不抽取关系
                try:
                    # 抽取实体，传递文件索引
                    entities = self._extract_entities(page.get("content", ""), page_number, chapter, file_index)

                    # 保存该页的实体
                    all_entities.extend(entities)
                    all_page_entities[page_number] = entities

                    logger.info(f"Extracted {len(entities)} entities from page {page_number}")
                except Exception as e:
                    logger.error(f"Error extracting entities from page {page_number}: {e}")
                    all_page_entities[page_number] = []

        # 删除current_title属性
        if hasattr(self, 'current_title'):
            delattr(self, 'current_title')

        # 第二步：按标题分组抽取关系
        all_relationships = []

        # 创建TextPreprocessor实例用于分组
        text_preprocessor = TextPreprocessor()
        grouped_pages = text_preprocessor.group_pages_by_title(preprocessed_data)

        logger.info(f"按标题将{len(preprocessed_data.get('pages', []))}页分成{len(grouped_pages)}组")

        # 遍历每个分组提取关系
        for group in grouped_pages:
            group_title = group.get("group_title", "")
            page_numbers = group.get("page_numbers", [])
            combined_content = group.get("combined_content", "")

            # 收集该组所有页面的实体
            group_entities = []
            for page_number in page_numbers:
                if page_number in all_page_entities:
                    group_entities.extend(all_page_entities[page_number])

            if not group_entities:
                logger.warning(f"Group with title '{group_title}' has no entities, skipping relationship extraction")
                continue

            logger.info(f"从分组'{group_title}'（{len(page_numbers)}页，{len(group_entities)}个实体）提取关系")

            try:
                # 从组合内容中提取关系，传递文件索引
                group_relationships = self._extract_relationships_from_group(
                    combined_content,
                    group_entities,
                    group_title,
                    chapter,
                    file_index
                )

                # 收集关系
                all_relationships.extend(group_relationships)
                logger.info(f"从分组'{group_title}'提取了{len(group_relationships)}个关系")
            except Exception as e:
                logger.error(f"Error extracting relationships from group '{group_title}': {e}")

        # 删除file_index属性
        if hasattr(self, 'file_index'):
            delattr(self, 'file_index')

        # 第三步：去重实体和关系
        logger.info(f"\nProcessing {len(all_entities)} entities for deduplication")
        unique_entities = self._remove_duplicate_entities(all_entities)
        logger.info(f"\nDeduplication complete: {len(all_entities)} entities reduced to {len(unique_entities)} unique entities")

        unique_relationships = self._remove_duplicate_relationships(all_relationships)

        # 更新输出
        output["entities"] = unique_entities
        output["relationships"] = unique_relationships

        # 生成统计信息
        entity_types = {}
        for entity in unique_entities:
            entity_type = entity["type"]
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        logger.info("\n\nExtraction statistics:")
        logger.info(f"Total entities: {len(unique_entities)}")
        logger.info(f"Entity types: {entity_types}")
        logger.info(f"Total relationships: {len(unique_relationships)}")

        return output

    def _process_page(self, page_data: Dict[str, Any], chapter: str, file_index: int = 0) -> Dict[str, Any]:
        """
        Process a single page to extract entities and relationships.

        Args:
            page_data: Dictionary containing page data
            chapter: Chapter information
            file_index: File index for creating globally unique entity IDs

        Returns:
            Dictionary containing extracted entities and relationships for the page
        """
        page_number = page_data["page_number"]
        # 使用 content 字段替代 full_text（适配简化版输出）
        page_text = page_data.get("content", page_data.get("full_text", ""))

        # Extract entities
        entities = self._extract_entities(page_text, page_number, chapter, file_index)

        # Extract relationships - with better error handling
        try:
            relationships = self._extract_relationships(page_text, entities, page_number, chapter, file_index)
        except Exception as e:
            logger.error(f"Error extracting relationships for page {page_number}: {str(e)}")
            # Return empty relationships if there's an error, but keep the entities
            relationships = []

        logger.info(f"Page {page_number} processed: {len(entities)} entities, {len(relationships)} relationships")

        return {
            "page_number": page_number,
            "entities": entities,
            "relationships": relationships
        }

    def _extract_entities(self, text: str, page_number: int, chapter: str, file_index: int) -> List[Dict[str, Any]]:
        """
        Extract entities from text.

        Args:
            text: Input text
            page_number: Page number
            chapter: Chapter information
            file_index: File index for creating globally unique entity IDs

        Returns:
            List of extracted entities
        """
        # Log the input text
        logger.info(f"\nProcessing page {page_number}:")
        logger.info(f"Input text length: {len(text)}")
        logger.info(f"First 200 chars: {text[:200]}...")

        # Check if the text appears to be a roadmap, TOC, or chapter introduction
        is_roadmap_or_toc = self._is_roadmap_or_toc(text)
        if is_roadmap_or_toc:
            logger.info(f"Page {page_number} appears to be a roadmap, TOC, or chapter introduction")

        # Use title from preprocessed data if available, otherwise extract from text
        if hasattr(self, 'current_title') and self.current_title:
            context = self.current_title
            logger.info(f"Using title from preprocessed data: {context}")
        else:
            # Extract context information from the text (title or a brief summary)
            context = self._extract_context_from_text(text)
            logger.info(f"Extracted context from text: {context}")

        # Prepare prompt for entity extraction
        prompt = self.entity_prompt.format(
            text=text,
            entity_types=", ".join(config.ENTITY_TYPES)
        )

        # Log the prompt
        logger.info(f"\nEntity extraction prompt:")
        logger.info(prompt)

        try:
            # Call LLM API
            response = self.llm_client.call(prompt)

            # Log the raw response
            logger.info(f"\nAPI Raw Response for page {page_number}:")
            logger.info(f"Response content: {response}")

            # Parse JSON response
            try:
                # 尝试解析响应结果
                try:
                    entities_data = json.loads(response)
                except json.JSONDecodeError:
                    # 如果直接解析失败，尝试修复常见的JSON问题
                    fixed_response = self._fix_json_response(response)
                    entities_data = json.loads(fixed_response)
                    logger.info(f"Fixed JSON response: {fixed_response[:200]}...")

                # 如果响应不是列表，尝试从中提取列表
                if not isinstance(entities_data, list):
                    logger.warning(f"API response is not a list, attempting to fix: {type(entities_data)}")

                    # 如果响应是字典，检查是否有可能包含实体列表的字段
                    if isinstance(entities_data, dict):
                        for key in ["entities", "entity_list", "results", "items", "data"]:
                            if key in entities_data and isinstance(entities_data[key], list):
                                entities_data = entities_data[key]
                                logger.info(f"Extracted entities list from '{key}' field")
                                break

                    # 如果仍然不是列表，则转换成空列表
                    if not isinstance(entities_data, list):
                        logger.error(f"Failed to extract list from response, using empty list")
                        entities_data = []

                # Format entities with additional filtering
                entities = []
                for i, entity_data in enumerate(entities_data):
                    if not isinstance(entity_data, dict):
                        logger.warning(f"Invalid entity data: {entity_data}")
                        continue

                    # Handle both "name" and "entity" fields
                    entity_name = entity_data.get("name") or entity_data.get("entity")
                    if not entity_name:
                        logger.warning(f"Entity data missing name/entity field: {entity_data}")
                        continue

                    # Skip if entity name matches exclusion patterns
                    if self._should_exclude_entity(entity_name):
                        logger.info(f"Skipping non-course entity: {entity_name}")
                        continue

                    # Get function type and set weight
                    function_type = entity_data.get("function", "mention")

                    # If the page is a roadmap/TOC and the function is "mention", change it to "title"
                    if is_roadmap_or_toc and function_type == "mention":
                        function_type = "title"
                        logger.info(f"Changed function type for '{entity_name}' from 'mention' to 'title' because the page appears to be a roadmap/TOC")

                    weight = {
                        "title": 1,
                        "mention": 2,
                        "introduction": 3
                    }.get(function_type, 2)

                    # Create occurrence information
                    filename = self.current_filename if hasattr(self, 'current_filename') else "unknown"
                    occurrence = [
                        f"{filename} page{page_number}",
                        function_type,
                        weight
                    ]

                    entity = {
                        "id": f"entity_{file_index}_{page_number}_{i+1}",
                        "name": entity_name,
                        "type": entity_data.get("type", "concept"),
                        "page_number": page_number,
                        "chapter": chapter,
                        "first_occurrence": True,
                        "occurrences": [occurrence],
                        "context": context  # 添加上下文信息
                    }

                    entities.append(entity)
                    logger.info(f"Created entity: {entity}")

                return entities

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response: {response}")
                return []

        except Exception as e:
            logger.error(f"\nError extracting entities for page {page_number}:")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error message: {str(e)}")
            return []

    def _extract_context_from_text(self, text: str) -> str:
        """
        Extract context information from text - generally the title or a brief summary of content.

        Args:
            text: Text to extract context from

        Returns:
            Context string
        """
        # 先尝试从文本中提取标题
        # 1. 检查首行，通常包含标题
        lines = text.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # 如果首行看起来像标题（不太长，不以常见的非标题词开头）
            if len(first_line) < 100 and not any(first_line.lower().startswith(x) for x in ['the ', 'a ', 'an ', 'in ', 'on ', 'for ', 'with ']):
                return first_line

        # 2. 查找常见标题模式
        title_patterns = [
            r'^([^\n.]{5,80})[\n:]',  # 以换行或冒号结尾的开头段落
            r'#\s+([^\n]{5,80})',     # Markdown风格的标题
            r'^\s*\*\*([^\n*]{5,80})\*\*',  # 粗体文本作为标题
            r'^\s*([A-Z][^a-z\n]{4,79})',   # 全大写的行
            r'^\s*([A-Z][a-zA-Z0-9 ]{4,79})(?:\n|\.|$)'  # 首字母大写的短行
        ]

        for pattern in title_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        # 3. 如果没有明确的标题，尝试提取前50个单词作为摘要
        words = text.split()
        if len(words) > 10:
            summary = ' '.join(words[:min(50, len(words))])
            # 如果摘要太长，可能不是标题，尝试缩短到句号或其他标点符号
            if len(summary) > 100:
                first_sentence = re.split(r'[.!?]', summary)[0]
                if len(first_sentence) > 10:
                    return first_sentence + '.'
            return summary

        # 4. 如果都不适用，返回一个通用的上下文
        return "General computer network content"

    def _is_roadmap_or_toc(self, text: str) -> bool:
        """
        Check if the text appears to be a roadmap, table of contents, or chapter introduction.

        Args:
            text: Text to analyze

        Returns:
            True if the text appears to be a roadmap, TOC, or chapter introduction, False otherwise
        """
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()

        # Patterns that suggest the text is a roadmap or TOC
        roadmap_patterns = [
            r'topics\s*:',
            r'our\s+goal\s*:',
            r'what\s+we\'ll\s+learn',
            r'what\s+we\s+will\s+learn',
            r'in\s+this\s+chapter\s+we\s+will\s+cover',
            r'in\s+this\s+course\s+we\s+will\s+cover',
            r'chapter\s+\d+\s*:\s*introduction',
            r'roadmap',
            r'outline',
            r'overview',
            r'syllabus',
            r'objectives',
            r'learning\s+outcomes',
            r'table\s+of\s+contents',
            r'course\s+outline',
            r'chapter\s+overview',
            r'contents'
        ]

        # Check for bullet points or numbered lists (at least 3 to be significant)
        bullet_pattern = re.compile(r'(?:\n\s*[•\-\*\®]\s+.+){3,}')
        numbered_pattern = re.compile(r'(?:\n\s*\d+[\.\)]\s+.+){3,}')

        # Check for the presence of patterns
        for pattern in roadmap_patterns:
            if re.search(pattern, text_lower):
                return True

        # Check for bullet points or numbered lists
        if bullet_pattern.search(text_lower) or numbered_pattern.search(text_lower):
            return True

        return False

    def _fix_json_response(self, response: str) -> str:
        """
        Fix common JSON formatting issues in API responses.

        Args:
            response: Raw API response

        Returns:
            Fixed JSON string
        """
        # 去除可能导致解析错误的不可见字符
        response = response.strip()

        # 如果响应为空，返回空列表
        if not response:
            return "[]"

        # 尝试提取JSON代码块
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, response)
        if match:
            response = match.group(1).strip()

        # 检查是否需要添加方括号使其成为JSON数组
        response = response.strip()
        if not response.startswith('[') and not response.startswith('{'):
            response = '[' + response + ']'

        # 修复常见格式问题
        response = response.replace("'", '"')  # 将单引号替换为双引号
        response = re.sub(r',\s*([}\]])', r'\1', response)  # 删除对象和数组末尾的逗号
        response = re.sub(r'(\w+):', r'"\1":', response)  # 将非引号包裹的键名添加引号

        # 修复未闭合的括号和引号
        open_braces = response.count('{')
        close_braces = response.count('}')
        open_brackets = response.count('[')
        close_brackets = response.count(']')

        # 添加缺少的右花括号
        if open_braces > close_braces:
            response += '}' * (open_braces - close_braces)

        # 添加缺少的右方括号
        if open_brackets > close_brackets:
            response += ']' * (open_brackets - close_brackets)

        # 检查是否有奇数个双引号，这可能表示未闭合的字符串
        if response.count('"') % 2 != 0:
            # 尝试通过添加额外的双引号来修复
            response += '"'

        return response

    def _extract_relationships(self, text: str, entities: List[Dict[str, Any]],
                              page_number: int, chapter: str, file_index: int = 0) -> List[Dict[str, Any]]:
        """
        Extract relationships from text.

        Args:
            text: Text to extract relationships from
            entities: List of entities extracted from the text
            page_number: Page number of the text
            chapter: Chapter of the text
            file_index: File index for creating globally unique relationship IDs

        Returns:
            List of extracted relationships
        """
        # If no entities were found, no relationships can exist
        if not entities:
            return []

        # Format entities for the prompt
        entities_str = json.dumps([{k: v for k, v in entity.items() if k in ["name", "type"]} for entity in entities],
                                 indent=2, ensure_ascii=False)

        # Prepare prompt for relationship extraction
        prompt = self._load_prompt_template("relationship_extraction").format(
            text=text,
            entities=entities_str
        )

        # Call LLM API
        response = self.llm_client.call(prompt)

        # Parse the response to extract relationships
        relationships = self._parse_relationships(response, entities, page_number, chapter, file_index)

        # Generate unique IDs for relationships
        for i, relationship in enumerate(relationships):
            relationship["id"] = f"rel_{file_index}_{page_number}_{i+1}"

        return relationships

    def _parse_relationships(self, response: str, entities: List[Dict[str, Any]],
                            page_number: int, chapter: str, file_index: int = 0) -> List[Dict[str, Any]]:
        """
        Parse relationships from LLM response.

        Args:
            response: LLM response
            entities: List of entities
            page_number: Page number
            chapter: Chapter
            file_index: File index for creating globally unique IDs

        Returns:
            List of parsed relationships
        """
        relationships = []

        # Create entity name to ID mapping for faster lookups
        entity_name_to_id = {}
        for entity in entities:
            entity_name_to_id[entity["name"]] = entity["id"]

        # Fix JSON response if needed
        response = self._fix_json_response(response)

        try:
            # Parse JSON response
            data = json.loads(response)

            # 如果响应是字典且包含entity_pairs字段，提取实体对列表
            if isinstance(data, dict) and "entity_pairs" in data:
                logger.info("Detected entity_pairs field in response, extracting entity pairs list")
                data = data["entity_pairs"]

            if not isinstance(data, list):
                logger.warning(f"Expected list in response, got {type(data)}")

                # 尝试从字典中提取可能的实体列表字段
                if isinstance(data, dict):
                    for key in ["entity_pairs", "relationships", "pairs", "result", "results"]:
                        if key in data and isinstance(data[key], list):
                            data = data[key]
                            logger.info(f"Extracted relationships list from '{key}' field")
                            break

                # 如果仍然不是列表，返回空结果
                if not isinstance(data, list):
                    logger.warning("Could not find a valid entity pairs list in the response")
                    return []

            # Process each entity pair
            for i, pair_data in enumerate(data):
                if not isinstance(pair_data, dict):
                    logger.warning(f"Expected dict for entity pair, got {type(pair_data)}")
                    continue

                # Get entity names
                entity1_name = pair_data.get("entity1", "")
                entity2_name = pair_data.get("entity2", "")

                # Ensure names are strings
                entity1_name = str(entity1_name) if entity1_name else ""
                entity2_name = str(entity2_name) if entity2_name else ""

                # 跳过自环关系（entity1与entity2相同）
                if entity1_name == entity2_name:
                    logger.warning(f"Skipping self-referential relationship: {entity1_name} -> {entity2_name}")
                    continue

                # Check if both entities exist
                if entity1_name in entity_name_to_id and entity2_name in entity_name_to_id:
                    # Get entity IDs
                    entity1_id = entity_name_to_id[entity1_name]
                    entity2_id = entity_name_to_id[entity2_name]

                    # 再次检查，确保ID不同
                    if entity1_id == entity2_id:
                        logger.warning(f"Skipping self-referential relationship: {entity1_name}({entity1_id}) -> {entity2_name}({entity2_id})")
                        continue

                    # Create relationship dictionary (without direction)
                    relationship = {
                        "entity1": entity1_id,
                        "entity2": entity2_id,
                        "entity1_name": entity1_name,
                        "entity2_name": entity2_name,
                        "page_number": page_number,
                        "chapter": chapter,
                        "type": "related_to"  # Default type, will be determined later
                    }

                    relationships.append(relationship)
                else:
                    if entity1_name not in entity_name_to_id:
                        logger.warning(f"Entity1 not found: {entity1_name}")
                    if entity2_name not in entity_name_to_id:
                        logger.warning(f"Entity2 not found: {entity2_name}")

        except Exception as e:
            logger.error(f"Error parsing entity pairs: {e}")
            logger.error(f"Response: {response[:500]}...")
            return []

        # Always check for isolated entities and add default connections
        # rather than checking if relationships < entities / 2
        self._add_default_relationships(relationships, entities, entity_name_to_id, page_number, chapter, file_index)

        return relationships

    def _add_default_relationships(self, relationships: List[Dict[str, Any]], entities: List[Dict[str, Any]],
                                  entity_name_to_id: Dict[str, str], page_number: int, chapter: str, file_index: int = 0) -> None:
        """
        Add default relationships for isolated entities.

        Args:
            relationships: List of relationships
            entities: List of entities
            entity_name_to_id: Mapping of entity names to IDs
            page_number: Page number
            chapter: Chapter
            file_index: File index for creating globally unique relationship IDs
        """
        # Count relationships for each entity
        entity_relationship_count = {}

        # Initialize counts
        for entity in entities:
            entity_relationship_count[entity["id"]] = 0

        # Count relationships per entity
        for rel in relationships:
            entity_relationship_count[rel["entity1"]] = entity_relationship_count.get(rel["entity1"], 0) + 1
            entity_relationship_count[rel["entity2"]] = entity_relationship_count.get(rel["entity2"], 0) + 1

        # Find isolated entities
        isolated_entities = [entity for entity in entities if entity_relationship_count.get(entity["id"], 0) == 0]

        if not isolated_entities:
            return

        logger.info(f"Found {len(isolated_entities)} isolated entities. Adding default connections.")

        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            entity_type = entity["type"]
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)

        # Connect isolated entities
        for isolated_entity in isolated_entities:
            # Try to find entities on same page
            same_page_entities = [e for e in entities if e["page_number"] == isolated_entity["page_number"] and e["id"] != isolated_entity["id"]]

            # Connect to an entity on the same page
            if same_page_entities:
                partner_entity = same_page_entities[0]

                # Create default relationship
                rel_id = f"rel_{file_index}_{page_number}_{len(relationships) + 1}"

                relationship = {
                    "id": rel_id,
                    "entity1": isolated_entity["id"],
                    "entity2": partner_entity["id"],
                    "entity1_name": isolated_entity["name"],
                    "entity2_name": partner_entity["name"],
                    "page_number": page_number,
                    "chapter": chapter,
                    "type": "related_to",
                    "metadata": {
                        "auto_generated": True,
                        "reason": "isolated_entity_default_connection"
                    }
                }

                relationships.append(relationship)
                logger.info(f"Added default connection for isolated entity: {isolated_entity['name']} -> {partner_entity['name']}")
            else:
                # If no entities on the same page, connect to any other entity
                other_entities = [e for e in entities if e["id"] != isolated_entity["id"]]
                if other_entities:
                    partner_entity = other_entities[0]

                    # Create default relationship
                    rel_id = f"rel_{file_index}_{page_number}_{len(relationships) + 1}"

                    relationship = {
                        "id": rel_id,
                        "entity1": isolated_entity["id"],
                        "entity2": partner_entity["id"],
                        "entity1_name": isolated_entity["name"],
                        "entity2_name": partner_entity["name"],
                        "page_number": page_number,
                        "chapter": chapter,
                        "type": "related_to",
                        "metadata": {
                            "auto_generated": True,
                            "reason": "isolated_entity_default_connection"
                        }
                    }

                    relationships.append(relationship)
                    logger.info(f"Added default connection for isolated entity: {isolated_entity['name']} -> {partner_entity['name']}")

    def _remove_duplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate entities based on name.

        Args:
            entities: List of entities

        Returns:
            List of unique entities
        """
        logger.info(f"\nProcessing {len(entities)} entities for deduplication")

        unique_entities = {}

        for entity in entities:
            name = entity["name"].lower()

            if name not in unique_entities:
                # Mark as first occurrence
                entity["first_occurrence"] = True

                # Copy occurrence list - already formatted with function and weight
                unique_entities[name] = entity
                logger.info(f"New unique entity: {entity['name']} (occurrences: {entity['occurrences']}, context: {entity.get('context', 'None')})")
            else:
                # Update existing entity with additional occurrences
                for occurrence in entity["occurrences"]:
                    if occurrence not in unique_entities[name]["occurrences"]:
                        unique_entities[name]["occurrences"].append(occurrence)
                        logger.info(f"Added occurrence of {entity['name']}: {occurrence}")

                # If this entity has a definition and the existing one doesn't, use this definition
                if "definition" in entity and "definition" not in unique_entities[name]:
                    unique_entities[name]["definition"] = entity["definition"]
                    logger.info(f"Added definition for {entity['name']} from page {entity['page_number']}")

                # 收集上下文信息，如果本实体有上下文信息
                if "context" in entity and entity["context"]:
                    if "context_list" not in unique_entities[name]:
                        unique_entities[name]["context_list"] = []

                    # 保存上下文和对应的权重（取自occurrence）
                    max_weight = max([occ[2] for occ in entity["occurrences"]]) if entity["occurrences"] else 0
                    context_entry = {
                        "context": entity["context"],
                        "weight": max_weight,
                        "page_number": entity["page_number"]
                    }
                    unique_entities[name]["context_list"].append(context_entry)
                    logger.info(f"Added context for {entity['name']} from page {entity['page_number']}: {entity['context']}")

        # Set chapter and primary context based on the occurrence with highest weight
        for entity_name, entity in unique_entities.items():
            if "occurrences" in entity and entity["occurrences"]:
                # Sort by weight (index 2) descending, then by occurrence string for stability
                sorted_occurrences = sorted(entity["occurrences"], key=lambda x: (-x[2], x[0]))

                # Get the highest weighted occurrence
                highest_weighted_occurrence = sorted_occurrences[0]

                # Update chapter information if we can extract it from occurrence
                occurrence_parts = highest_weighted_occurrence[0].split()
                if len(occurrence_parts) >= 1:
                    filename = occurrence_parts[0]
                    if "chapter" in entity:
                        # Extract chapter number if possible
                        chapter_match = re.search(r'chap(\d+)', filename, re.IGNORECASE)
                        if chapter_match:
                            chapter_num = chapter_match.group(1)
                            entity["chapter"] = f"Chapter {chapter_num}"
                            logger.info(f"Updated chapter for {entity['name']} to {entity['chapter']} based on weighted occurrence")

            # 为实体设置主要上下文，基于最高权重的上下文
            if "context_list" in entity:
                # 按权重降序排序上下文列表
                sorted_contexts = sorted(entity["context_list"], key=lambda x: (-x["weight"], x["page_number"]))

                # 使用权重最高的上下文作为实体的主要上下文
                if sorted_contexts:
                    entity["context"] = sorted_contexts[0]["context"]
                    logger.info(f"Selected primary context for {entity['name']}: {entity['context']}")

                # 保留完整的上下文列表以供参考
                # 但也可以删除，取决于是否需要保留所有上下文
                # del entity["context_list"]

        result = list(unique_entities.values())

        logger.info(f"\nDeduplication complete: {len(entities)} entities reduced to {len(result)} unique entities")
        for entity in result:
            logger.info(f"- {entity['name']} (type: {entity['type']}): appears at {entity['occurrences']}, context: {entity.get('context', 'None')}")

        return result

    def _remove_duplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate relationships.

        Args:
            relationships: List of relationships

        Returns:
            List of unique relationships
        """
        # 使用集合来存储唯一关系键（entity1_entity2）
        unique_keys = set()
        unique_relationships = []

        # 记录遇到的自环关系数量
        self_loop_count = 0

        for relationship in relationships:
            # 检查是否为自环关系（entity1和entity2相同）
            if relationship["entity1"] == relationship["entity2"]:
                logger.warning(f"Removing self-referential relationship: {relationship['entity1_name']} -> {relationship['entity2_name']}")
                self_loop_count += 1
                continue

            # 使用source和target作为唯一键，不再依赖type
            key = f"{relationship['entity1']}_{relationship['entity2']}"

            if key not in unique_keys:
                unique_keys.add(key)
                unique_relationships.append(relationship)
                # 不再使用或记录page_number作为出现位置

        if self_loop_count > 0:
            logger.info(f"已移除 {self_loop_count} 个自环关系")

        logger.info(f"关系去重: {len(relationships)} 减少到 {len(unique_relationships)} 个唯一关系")
        return unique_relationships

    def _load_prompt_template(self, template_name: str) -> str:
        """
        Load a prompt template.

        Args:
            template_name: Name of the template

        Returns:
            Prompt template string
        """
        # In a real implementation, you would load templates from files

        if template_name == "entity_extraction":
            return """
            You are a knowledgeable computer network domain expert.
            Extract entities (concepts, protocols, physical media, technologies, etc.) from the following text from a computer network course slide.

            Text:
            ```
            {text}
            ```

            Return entities in the following JSON format:
            [
                {{
                    "name": "Entity name (use the exact form from the text)",
                    "type": "One of: concept, protocol, physical_media, technology, other",
                    "function": "One of:title,mention,introduction"

                }}
            ]
            Example response format:
            [
                {{
                    "name": "computer network",
                    "type": "concept",
                    "function": "introduction"
                }},
                {{
                    "name": "protocol Layers",
                    "type": "concept",
                    "function": "mention"
                }},
                {{
                    "name": "data transmission",
                    "type": "concept",
                    "function": "mention"
                }}
            ]

            Guidelines:
            1. VERY IMPORTANT: Be sure to extract fundamental computer network concepts such as:
               - "computer network" itself
               - "protocol layers"
               - "service models"
               - "data transmission"
               - "Wi-Fi"
               These are critical core concepts that must be extracted even if they only appear briefly.

            2. ONLY extract entities that are SPECIFICALLY related to computer networks. Ignore:
               - Course codes (e.g., CS339)
               - University names
               - Course administrative terms
               - General computing concepts not specific to networking
               - People names
               - Dates and times
               - Page numbers and section numbers

            3. DO NOT extract specific numerical values and technical details:
               - DO NOT extract specific IP addresses (e.g., 192.168.1.1)
               - DO NOT extract specific frequency values (e.g., 2170 MHz, 5GHz)
               - DO NOT extract specific bit rates (e.g., 1.5 Mbps, 28.8 Kbps)
               - DO NOT extract specific version numbers (e.g., v1.2)
               - DO NOT extract specific standards numbers (e.g., 802.11g)
               - DO NOT extract specific RFC numbers (e.g., RFC 2326)
               - DO NOT extract URLs or website addresses

            4. Instead of specific values, extract general concepts:
               - Extract "IP address" instead of "192.168.1.1"
               - Extract "frequency band" instead of "2170 MHz"
               - Extract "bandwidth" instead of "1.5 Mbps"
               - Extract "IEEE 802.11 standard" instead of "802.11g"
               - Extract "RFC standard" instead of "RFC 2326"

            5. Extract the full entity name as it appears in the text (e.g., "Transmission Control Protocol" not just "TCP").
            6. If an acronym appears alongside its expansion, extract both (e.g., "TCP (Transmission Control Protocol)").
            7. Classify entities into one of the following types:
               - concept: Fundamental networking concepts and principles
               - protocol: Network protocols and standards
               - physical_media: Physical transmission media and hardware
               - technology: Networking technologies and implementations
               - other: Other networking-specific entities
            8. If unsure about an entity type, use "other".
            9. For each knowledge point, CAREFULLY determine its function in this text as one of:
               - "title": The concept just appears in a table of contents, roadmap, syllabus, chapter introduction, outline or heading without any explanation. This includes when concepts are just listed as bullet points or in a catalog/overview of what will be covered in the course or chapter.
               - "mention": The concept is briefly mentioned in the main content but not extensively explained
               - "introduction": The text primarily explains this concept in detail or with examples
            10. Pay SPECIAL ATTENTION to the context:
               - If a term appears in an introduction slide, chapter summary, or course overview, mark it as "title"
               - If the text contains phrases like "topics:", "our goal:", "what we'll learn:", "in this chapter/course we will cover:", or similar introductory statements followed by lists of concepts, mark those concepts as "title"
               - Only use "mention" when the concept appears in the main explanatory content but isn't the focus
               - Use "introduction" when significant explanation is provided about the concept
            11. For entity names, keep acronyms as they appear (e.g., "OSI model", "TCP/IP").
            12. Focus on technical terms and concepts rather than common words.
            13. Ensure the JSON is valid - use double quotes for strings and proper nesting.

            Remember to only return a valid JSON array of entity objects that strictly follows the format above.
            IMPORTANT:Avoid extracting specific numerical values and technical details as mentioned above.Do NOT miss any core networking concepts, even if they are only mentioned briefly.
            """
        elif template_name == "relationship_extraction":
            return """
            You are a computer network course instructor. Your task is to identify entity pairs that have meaningful educational relationships in computer network course material.
            Identify related entity pairs from the following text and the extracted entities.

            Text:
            ```
            {text}
            ```

            Extracted Entities:
            {entities}

            Return entity pairs in the following JSON format:
            [
                {{
                    "entity1": "Entity name 1",
                    "entity2": "Entity name 2"
                }}
            ]

            Guidelines:
            1. ONLY create pairs between entities that are in the provided list of extracted entities.
            2. EDUCATIONAL FOCUS: Create as many MEANINGFUL pairs as possible between entities. It's CRITICAL that EVERY entity has at least one relationship.
            3. Use the EXACT entity names as they appear in the extracted entities list.
            4. Create pairs even when not explicitly stated in the text - use your domain knowledge to connect related concepts.
            5. QUANTITY IS IMPORTANT: Aim to create at least 2-3 relationships for each entity in the list.
            6. Include BOTH strongly AND weakly related pairs. Even a tangential connection can be valuable for learning.
            7. For each pair of entities, consider ALL possible educational connections that would help a student understand their relationship.
            8. DO NOT include any relationship type, direction, or definition in your response - just identify the entity pairs.

            Additional guidelines:
            1. Think about concepts that are hierarchically, temporally, or functionally related.
            2. Connect concepts that appear on the same page - they are likely related in some way.
            3. If a concept seems isolated, make EXTRA EFFORT to find meaningful connections.
            4. ABSOLUTELY AVOID leaving any entity without connections - isolated knowledge is difficult to remember.
            5. For technical terms, connect them to more general concepts they belong to.
            6. For acronyms, connect them to their expanded forms and related concepts.

            Remember to only return a valid JSON array of entity pair objects that strictly follows the format above.
            Focus on QUANTITY and COVERAGE - ensure EVERY entity has multiple relationships.
            """
        else:
            return ""

    def _should_exclude_entity(self, entity_name: str) -> bool:
        """
        Check if an entity should be excluded based on various rules.

        Args:
            entity_name: Name of the entity to check

        Returns:
            True if the entity should be excluded, False otherwise
        """
        # Convert to lowercase for case-insensitive matching
        name_lower = entity_name.lower()

        # 显式保留的核心概念列表（这些概念永远不会被过滤）
        core_concepts = {
            'computer network', 'network', 'protocol', 'internet', 'protocol layer',
            'service model', 'packet switching', 'circuit switching', 'tcp/ip',
            'osi model', 'routing', 'bandwidth', 'delay', 'throughput',
            'multiplexing', 'switching', 'network edge', 'network core',
            'physical media', 'transmission', 'data transmission','internet','Wi-Fi'
        }

        # 如果是核心概念，直接返回False（不排除）
        if name_lower in core_concepts:
            return False

        # Patterns for exclusion
        exclusion_patterns = [
            r'^[a-zA-Z]$',  # Single letters
            r'@',  # Email addresses
            r'\.edu\.',  # Educational institution domains
            r'\.com$',
            r'\.cn$',
            r'\.org$',
            r'chapter\s+\d+',  # Chapter numbers
            r'^\d+$',  # Just numbers
            r'page\s+\d+',  # Page numbers
            r'^(january|february|march|april|may|june|july|august|september|october|november|december)$',  # Months
            r'^(mon|tue|wed|thu|fri|sat|sun)$',  # Days
            r'^\d{4}$',  # Years
            r'^v\d+(\.\d+)*$',  # Version numbers
            r'^[A-Z]{2,3}\d{3}$',  # Course codes like CS339
            r'^[A-Z]{2,3}\s*\d{3}$',  # Course codes with space
            r'university$',  # University names
            r'college$',  # College names
            r'school$',  # School names
            r'institute$',  # Institute names
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+$',  # Likely university names with three words
            # 新增：IP地址相关模式
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',  # 具体的IPv4地址
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',  # CIDR格式的IP地址
            r'^[0-9a-fA-F:]+/\d{1,3}$',  # IPv6 CIDR格式
            r'^[0-9a-fA-F:]+$',  # IPv6地址
            # 新增：日期相关模式
            r'^\d{4}/\d{1,2}/\d{1,2}$',  # YYYY/MM/DD格式
            r'^\d{1,2}/\d{1,2}/\d{4}$',  # MM/DD/YYYY格式
            r'^\d{4}-\d{1,2}-\d{1,2}$',  # YYYY-MM-DD格式
            r'^\d{1,2}-\d{1,2}-\d{4}$',  # DD-MM-YYYY格式
            r'^\d{1,2}\.\d{1,2}\.\d{4}$',  # DD.MM.YYYY格式
            # 新增：通用数字和符号模式
            r'^[\d\.\-/]+$',  # 仅包含数字、点、横线和斜杠的字符串
            r'^[a-z]\.[a-z]\.[a-z]\.[a-z]$',  # 类似 a.b.c.d 的模式
            r'^[a-z]\.[a-z]\.[a-z]\.[a-z]/[a-z]$'  # 类似 a.b.c.d/x 的模式
        ]

        # Common words to exclude (移除了可能与课程核心概念相关的词)
        common_words = {
            'introduction', 'overview', 'summary', 'conclusion', 'appendix',
            'figure', 'table', 'section', 'chapter', 'page', 'slide',
            'example', 'note', 'reference', 'bibliography', 'glossary',
            'content', 'index', 'list', 'title', 'header', 'footer',
            'lecture', 'tutorial', 'lab', 'assignment', 'homework',
            'professor', 'instructor', 'teacher', 'student', 'class',
            'semester', 'term', 'year', 'session', 'module'
        }

        # Check patterns
        for pattern in exclusion_patterns:
            if re.search(pattern, name_lower):
                return True

        # Check common words (但只有当名称完全等于某个common word时才排除)
        if name_lower in common_words and len(name_lower.split()) == 1:
            return True

        # Check if it's likely a person name (two capitalized words)
        if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', entity_name):
            return True

        # Check for university names (common patterns)
        if any(word in name_lower for word in ['university', 'college', 'school', 'institute']):
            return True

        return False

    def _extract_relationships_from_group(self, text: str, entities: List[Dict[str, Any]],
                                        group_title: str, chapter: str, file_index: int = 0) -> List[Dict[str, Any]]:
        """
        从按标题分组的页面中提取关系。

        Args:
            text: 分组页面的组合文本
            entities: 该组页面中的所有实体
            group_title: 分组标题
            chapter: 章节信息
            file_index: 文件索引号，用于创建全局唯一的实体ID

        Returns:
            提取的关系列表
        """
        # 日志记录
        logger.info(f"\n处理标题组: '{group_title}'")
        logger.info(f"实体数量: {len(entities)}")
        logger.info(f"文本长度: {len(text)}")
        logger.info(f"前200个字符: {text[:200]}...")

        # 如果没有实体，无法创建关系
        if not entities:
            return []

        # 格式化实体以用于提示
        entities_str = json.dumps([{k: v for k, v in entity.items() if k in ["name", "type"]} for entity in entities],
                                indent=2, ensure_ascii=False)

        # 准备关系提取提示
        prompt = self._load_prompt_template("relationship_extraction").format(
            text=text,
            entities=entities_str
        )

        # 调用LLM API
        response = self.llm_client.call(prompt)

        # 解析响应以提取关系
        entity_name_to_id = {entity["name"]: entity["id"] for entity in entities}
        relationships = []

        try:
            # 尝试解析响应结果
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                # 如果解析失败，尝试修复
                fixed_response = self._fix_json_response(response)
                data = json.loads(fixed_response)
                logger.info(f"Fixed relationship JSON response: {fixed_response[:200]}...")

            # 如果响应不是列表，尝试提取列表
            if not isinstance(data, list):
                logger.warning(f"Relationship API response is not a list, attempting to fix: {type(data)}")

                if isinstance(data, dict):
                    for key in ["relationships", "relationship_list", "results", "items", "data", "entity_pairs"]:
                        if key in data and isinstance(data[key], list):
                            data = data[key]
                            logger.info(f"Extracted relationships list from '{key}' field")
                            break

                if not isinstance(data, list):
                    logger.warning("Could not find a valid entity pairs list in the response")
                    return []

            # 处理每个关系
            for i, rel_data in enumerate(data):
                if not isinstance(rel_data, dict):
                    logger.warning(f"Expected dict for relationship, got {type(rel_data)}")
                    continue

                # 获取实体名称
                entity1_name = rel_data.get("entity1", "")
                entity2_name = rel_data.get("entity2", "")


                # 尝试处理可能的格式问题
                if isinstance(entity1_name, dict) and "name" in entity1_name:
                    entity1_name = entity1_name["name"]
                if isinstance(entity2_name, dict) and "name" in entity2_name:
                    entity2_name = entity2_name["name"]

                # 确保是字符串类型
                entity1_name = str(entity1_name) if entity1_name else ""
                entity2_name = str(entity2_name) if entity2_name else ""

                # 检查实体是否存在
                if entity1_name in entity_name_to_id and entity2_name in entity_name_to_id:
                    # 获取实体ID
                    entity1_id = entity_name_to_id[entity1_name]
                    entity2_id = entity_name_to_id[entity2_name]

                    # 跳过自环关系
                    if entity1_id == entity2_id:
                        logger.warning(f"Skipping self-referential relationship: {entity1_name}({entity1_id}) -> {entity2_name}({entity2_id})")
                        continue

                    # 提取关系类型，如果存在
                    rel_type = rel_data.get("type", "related_to")

                    # 创建关系字典（使用entity1/entity2格式）
                    relationship = {
                        "id": f"rel_{file_index}_group_{i+1}",
                        "entity1": entity1_id,
                        "entity2": entity2_id,
                        "entity1_name": entity1_name,
                        "entity2_name": entity2_name,
                        "chapter": chapter,
                        "type": rel_type
                    }

                    relationships.append(relationship)
                else:
                    if entity1_name not in entity_name_to_id:
                        logger.warning(f"Entity1 not found: {entity1_name}")
                    if entity2_name not in entity_name_to_id:
                        logger.warning(f"Entity2 not found: {entity2_name}")

        except Exception as e:
            logger.error(f"Error parsing relationships: {e}")
            logger.error(f"Relationship response: {response[:500]}...")
            return []

        # Always check for isolated entities and add default connections
        # rather than checking if relationships < entities / 2
        self._add_default_relationships_for_group(relationships, entities, entity_name_to_id, chapter, file_index)

        return relationships

    def _add_default_relationships_for_group(self, relationships: List[Dict[str, Any]], entities: List[Dict[str, Any]],
                                         entity_name_to_id: Dict[str, str], chapter: str, file_index: int = 0) -> None:
        """
        为按标题分组的孤立实体添加默认关系。

        Args:
            relationships: 现有关系列表
            entities: 实体列表
            entity_name_to_id: 实体名称到ID的映射
            chapter: 章节信息
            file_index: 文件索引号，用于创建全局唯一的实体ID
        """
        # 统计每个实体的关系数量
        entity_relationship_count = {}

        # 初始化计数
        for entity in entities:
            entity_relationship_count[entity["id"]] = 0

        # Count relationships per entity
        for rel in relationships:
            entity_relationship_count[rel["entity1"]] = entity_relationship_count.get(rel["entity1"], 0) + 1
            entity_relationship_count[rel["entity2"]] = entity_relationship_count.get(rel["entity2"], 0) + 1

        # Find isolated entities
        isolated_entities = [entity for entity in entities if entity_relationship_count.get(entity["id"], 0) == 0]

        if not isolated_entities:
            return

        logger.info(f"找到{len(isolated_entities)}个孤立实体")

        # 为孤立实体创建默认关系
        for i, isolated_entity in enumerate(isolated_entities):
            # 尝试找到最相关的实体
            best_match = None

            # 首先，尝试按类型匹配
            for entity in entities:
                if entity["id"] != isolated_entity["id"] and entity.get("type") == isolated_entity.get("type"):
                    best_match = entity
                    break

            # 如果没有找到相同类型的实体，使用第一个不同的实体
            if not best_match:
                for entity in entities:
                    if entity["id"] != isolated_entity["id"]:
                        best_match = entity
                        break

            # 如果找到了匹配的实体，创建关系
            if best_match:
                # 创建一个通用关系，类型为related_to
                relationship = {
                    "id": f"rel_{file_index}_group_default_{i+1}",
                    "entity1": isolated_entity["id"],
                    "entity2": best_match["id"],
                    "entity1_name": isolated_entity["name"],
                    "entity2_name": best_match["name"],
                    "chapter": chapter,
                    "type": "related_to"
                }

                relationships.append(relationship)
                logger.info(f"为孤立实体'{isolated_entity['name']}'创建了默认关系")
