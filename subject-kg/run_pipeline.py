#!/usr/bin/env python
"""
Pipeline class for running the Computer Network Course Knowledge Graph Construction pipeline.
"""

import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Any

# Import modules
from ocr.ocr_processor import OCRProcessor
from preprocessing.text_preprocessor import TextPreprocessor
from extraction.entity_extractor import EntityExtractor
from matching.entity_matcher import EntityMatcher
from definition.entity_definer import EntityDefiner
from exploration.type_explorer import TypeExplorer
from construction.kg_builder import KnowledgeGraphBuilder
from quality.quality_assessor import QualityAssessor
from course_db.db_manager import CourseDBManager
from canonicalization.relationship_canonicalizer import RelationshipCanonicalizer

# Import configuration
import config

# Set up logging
logger = logging.getLogger(__name__)

class RunPipeline:
    """
    Class for running the knowledge graph construction pipeline.
    """

    def __init__(self, model_name=config.DEFAULT_MODEL, api_key=None, api_base=None,
                ocr_dpi=config.OCR_DPI, poppler_path=None, tesseract_path=None,
                output_dir=config.OUTPUT_DIR):
        """
        Initialize the pipeline.

        Args:
            model_name: Name of the LLM model to use
            api_key: API key for the LLM service
            api_base: API base URL for the LLM service
            ocr_dpi: DPI for OCR processing
            poppler_path: Path to poppler binaries
            tesseract_path: Path to tesseract binary
            output_dir: Directory to save output files
        """
        self.model_name = model_name
        self.api_key = api_key or config.OPENAI_API_KEY
        self.api_base = api_base or config.OPENAI_API_BASE
        self.output_dir = output_dir

        # Set API key and base URL in environment variables
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if api_base:
            os.environ["OPENAI_API_BASE"] = api_base

        # Set Tesseract path
        if tesseract_path:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # Initialize modules
        self.ocr_processor = OCRProcessor(dpi=ocr_dpi, poppler_path=poppler_path)
        self.text_preprocessor = TextPreprocessor()
        self.entity_extractor = EntityExtractor(model_name=model_name, api_key=api_key, api_base=api_base)
        self.entity_matcher = EntityMatcher()
        self.entity_definer = EntityDefiner(model_name=model_name, api_key=api_key, api_base=api_base)
        self.type_explorer = TypeExplorer(model_name=model_name, api_key=api_key, api_base=api_base)
        self.kg_builder = KnowledgeGraphBuilder()
        self.quality_assessor = QualityAssessor()

        # Initialize course database manager
        db_dir = os.path.join(output_dir, "course_db")
        self.db_manager = CourseDBManager(db_dir)

        # Output directories are already created in config.py

        # Initialize statistics
        self.stats = {
            "pdfs_processed": 0,
            "pages_processed": 0,
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "entities_matched": 0,
            "relationships_matched": 0,
            "entities_defined": 0,
            "relationships_defined": 0
        }

        logger.info(f"Initialized pipeline with model {model_name}")

    def run(self, pdf_files: List[str], skip_ocr: bool = False, skip_preprocessing: bool = False,
            skip_extraction: bool = False, skip_matching: bool = False,
            skip_definition: bool = False, skip_canonicalization: bool = False,
            skip_exploration: bool = False, skip_quality: bool = False) -> None:
        """
        Process multiple PDF files in a batch-like manner, where each stage is completed for all files
        before moving to the next stage.

        Args:
            pdf_files: List of PDF file paths
            skip_ocr: Whether to skip OCR
            skip_preprocessing: Whether to skip preprocessing
            skip_extraction: Whether to skip extraction
            skip_matching: Whether to skip matching
            skip_definition: Whether to skip definition
            skip_canonicalization: Whether to skip canonicalization
            skip_exploration: Whether to skip exploration
            skip_quality: Whether to skip quality assessment
        """
        logger.info(f"Processing {len(pdf_files)} PDF files")

        # Initialize paths for combined outputs
        combined_matching_path = os.path.join(config.MATCHING_OUTPUT_DIR, "combined_matching.json")
        combined_definition_path = os.path.join(config.DEFINITION_OUTPUT_DIR, "combined_definition.json")

        # Create course_db directory if it doesn't exist
        course_db_dir = os.path.join(self.output_dir, "course_db")
        os.makedirs(course_db_dir, exist_ok=True)

        # Update paths to output to course_db directory
        combined_graph_path = os.path.join(course_db_dir, "combined_graph.json")
        quality_assessment_path = os.path.join(course_db_dir, "quality_assessment.json")
        entity_types_path = os.path.join(course_db_dir, "entity_types.json")
        relationship_types_path = os.path.join(course_db_dir, "relationship_types.json")

        # Step 1: OCR and preprocessing for all PDFs
        preprocessed_data_list = []
        for pdf_path in pdf_files:
            pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
            ocr_output_path = os.path.join(config.OCR_OUTPUT_DIR, f"{pdf_name}.txt")
            preprocessed_path = os.path.join(config.PREPROCESSED_OUTPUT_DIR, f"{pdf_name}.json")

            # OCR if needed
            if not skip_ocr and not os.path.exists(ocr_output_path):
                logger.info(f"Performing OCR on {pdf_path}...")
                ocr_output = self.ocr_processor.process(pdf_path)
                with open(ocr_output_path, 'w', encoding='utf-8') as f:
                    f.write(ocr_output)
                logger.info(f"OCR output saved to {ocr_output_path}")
            else:
                logger.info(f"Using existing OCR output for {pdf_path}")

            # Preprocessing if needed
            if not skip_preprocessing and not os.path.exists(preprocessed_path):
                with open(ocr_output_path, 'r', encoding='utf-8') as f:
                    ocr_output = f.read()
                logger.info(f"Preprocessing OCR output for {pdf_path}...")
                preprocessed_data = self.text_preprocessor.process(ocr_output)
                preprocessed_data["filename"] = pdf_name
                with open(preprocessed_path, 'w', encoding='utf-8') as f:
                    json.dump(preprocessed_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Preprocessed data saved to {preprocessed_path}")
            else:
                logger.info(f"Using existing preprocessed data for {pdf_path}")
                with open(preprocessed_path, 'r', encoding='utf-8') as f:
                    preprocessed_data = json.load(f)

            preprocessed_data_list.append((pdf_path, preprocessed_path, preprocessed_data))

            # Update statistics
            self.stats["pages_processed"] += len(preprocessed_data.get("pages", []))
            self.stats["pdfs_processed"] += 1

        # Step 2: Entity extraction and matching for all PDFs
        all_entities = []
        all_relationships = []

        for idx, (pdf_path, preprocessed_path, preprocessed_data) in enumerate(preprocessed_data_list):
            pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
            extraction_path = os.path.join(config.EXTRACTION_OUTPUT_DIR, f"{pdf_name}.json")
            matching_path = os.path.join(config.MATCHING_OUTPUT_DIR, f"{pdf_name}.json")

            # 每个文件使用唯一的file_index (从1开始)
            file_index = idx + 1

            # Extraction if needed
            if not skip_extraction and not os.path.exists(extraction_path):
                logger.info(f"Extracting entities and relationships for {pdf_path}... (file_index: {file_index})")
                # 传递file_index参数给extract方法
                extraction_data = self.entity_extractor.extract(preprocessed_data, file_index=file_index)
                with open(extraction_path, 'w', encoding='utf-8') as f:
                    json.dump(extraction_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Extraction data saved to {extraction_path}")
            else:
                logger.info(f"Using existing extraction data for {pdf_path}")
                with open(extraction_path, 'r', encoding='utf-8') as f:
                    extraction_data = json.load(f)

            # Update statistics
            self.stats["entities_extracted"] += len(extraction_data.get("entities", []))
            self.stats["relationships_extracted"] += len(extraction_data.get("relationships", []))

            # Matching
            if not skip_matching:
                logger.info(f"Matching entities for {pdf_path}...")
                # Load existing entities from the database to perform matching
                existing_entities = self.db_manager.load_entities()

                # 仅进行实体匹配，不重新编号
                matched_entities = self.entity_matcher.match_entities(
                    extraction_data.get("entities", []),
                    existing_entities
                )

                # 保存中间结果供调试
                matching_data = {
                    "filename": pdf_name,
                    "entities": matched_entities,
                    "relationships": extraction_data.get("relationships", [])
                }

                with open(matching_path, 'w', encoding='utf-8') as f:
                    json.dump(matching_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Matching data saved to {matching_path}")

                # 添加到合并列表中
                all_entities.extend(matched_entities)
                all_relationships.extend(extraction_data.get("relationships", []))
            else:
                logger.info(f"Using existing matching data for {pdf_path}")
                with open(matching_path, 'r', encoding='utf-8') as f:
                    matching_data = json.load(f)

                all_entities.extend(matching_data.get("entities", []))
                all_relationships.extend(matching_data.get("relationships", []))

            # Update statistics
            self.stats["entities_matched"] += len(matching_data.get("entities", []))
            self.stats["relationships_matched"] += len(matching_data.get("relationships", []))

        # 去重合并后的所有实体
        logger.info(f"去重前所有文件共有实体 {len(all_entities)} 个")

        # 创建实体名称到实体映射，用于去重
        entity_map_by_name = {}
        unique_entities = []

        for entity in all_entities:
            entity_name = entity["name"].lower()
            if entity_name in entity_map_by_name:
                # 已存在同名实体，合并occurrences
                existing_entity = entity_map_by_name[entity_name]

                # 确保occurrences列表格式正确
                if "occurrences" not in existing_entity:
                    existing_entity["occurrences"] = []
                if "occurrences" not in entity:
                    entity["occurrences"] = []

                # 合并occurrences，确保不重复
                existing_occurrences_set = {tuple(occ) for occ in existing_entity["occurrences"]}
                for occ in entity["occurrences"]:
                    existing_occurrences_set.add(tuple(occ))

                existing_entity["occurrences"] = [list(occ) for occ in existing_occurrences_set]

                # 保留更早的章节信息和首次出现信息
                if entity.get("first_occurrence", False) and not existing_entity.get("first_occurrence", False):
                    existing_entity["first_occurrence"] = True

                # 合并其他信息，如定义等
                if "definition" in entity and "definition" not in existing_entity:
                    existing_entity["definition"] = entity["definition"]

                # 如果新实体有更有意义的类型，使用新类型
                if entity.get("type") and entity.get("type") != "other" and (not existing_entity.get("type") or existing_entity.get("type") == "other"):
                    existing_entity["type"] = entity["type"]
            else:
                # 新实体，添加到列表
                unique_entities.append(entity)
                entity_map_by_name[entity_name] = entity

        logger.info(f"去重后剩余实体 {len(unique_entities)} 个")

        # 重新编号所有实体，从e0001开始
        renumbered_entities = []
        id_map = {}  # 旧ID到新ID的映射

        for i, entity in enumerate(unique_entities):
            old_id = entity["id"]
            new_id = f"e{i+1:04d}"  # 格式：e0001, e0002, ...

            # 更新实体ID
            entity["id"] = new_id
            id_map[old_id] = new_id
            renumbered_entities.append(entity)

        # 更新关系中的实体引用
        renumbered_relationships = []
        relationships_processed = set()  # 用于去重关系

        # 创建实体名称到新ID的映射
        entity_name_to_id = {}
        for entity in renumbered_entities:
            entity_name = entity["name"].lower()
            entity_name_to_id[entity_name] = entity["id"]

        # 记录处理前的关系数量
        logger.info(f"关系处理前: 总共 {len(all_relationships)} 个关系")

        for relationship in all_relationships:
            # 检查关系是否使用新格式(entity1/entity2)或旧格式(source/target)
            using_new_format = "entity1" in relationship and "entity2" in relationship

            if using_new_format:
                entity1_id = relationship.get("entity1")
                entity2_id = relationship.get("entity2")
                entity1_name = relationship.get("entity1_name", "").lower()
                entity2_name = relationship.get("entity2_name", "").lower()
            else:
                # 旧格式，使用source/target
                entity1_id = relationship.get("source")  # 视为entity1
                entity2_id = relationship.get("target")  # 视为entity2
                entity1_name = relationship.get("source_name", "").lower()
                entity2_name = relationship.get("target_name", "").lower()

            rel_type = relationship.get("type", "related_to")

            # 优先使用id_map更新关系的entity IDs
            new_entity1_id = None
            new_entity2_id = None

            # 1. 尝试使用旧ID映射
            if entity1_id in id_map:
                new_entity1_id = id_map[entity1_id]
            # 2. 没有找到，尝试通过name查找
            elif entity1_name and entity1_name in entity_name_to_id:
                new_entity1_id = entity_name_to_id[entity1_name]

            # 同样处理entity2
            if entity2_id in id_map:
                new_entity2_id = id_map[entity2_id]
            elif entity2_name and entity2_name in entity_name_to_id:
                new_entity2_id = entity_name_to_id[entity2_name]

            # 如果无法找到entity对应的实体，尝试在实体列表中查找
            if not new_entity1_id and entity1_name:
                for entity in renumbered_entities:
                    if entity["name"].lower() == entity1_name:
                        new_entity1_id = entity["id"]
                        break

            if not new_entity2_id and entity2_name:
                for entity in renumbered_entities:
                    if entity["name"].lower() == entity2_name:
                        new_entity2_id = entity["id"]
                        break

            # 如果仍然找不到，记录并跳过
            if not new_entity1_id:
                if using_new_format:
                    logger.warning(f"无法找到关系entity1实体: entity1_id={entity1_id}, entity1_name={entity1_name}")
                else:
                    logger.warning(f"无法找到关系source实体: source_id={entity1_id}, source_name={entity1_name}")
                continue

            if not new_entity2_id:
                if using_new_format:
                    logger.warning(f"无法找到关系entity2实体: entity2_id={entity2_id}, entity2_name={entity2_name}")
                else:
                    logger.warning(f"无法找到关系target实体: target_id={entity2_id}, target_name={entity2_name}")
                continue

            # 更新关系的ID引用
            new_relationship = relationship.copy()

            # 使用统一的新格式
            if using_new_format:
                new_relationship["entity1"] = new_entity1_id
                new_relationship["entity2"] = new_entity2_id
            else:
                # 转换旧格式为新格式
                new_relationship["entity1"] = new_entity1_id
                new_relationship["entity2"] = new_entity2_id
                # 保留source/target以兼容旧代码
                new_relationship["source"] = new_entity1_id
                new_relationship["target"] = new_entity2_id

            # 查找对应的实体名称，确保entity名称正确
            entity1 = next((e for e in renumbered_entities if e["id"] == new_entity1_id), None)
            entity2 = next((e for e in renumbered_entities if e["id"] == new_entity2_id), None)

            if entity1 and entity2:
                # 使用新格式的名称字段
                new_relationship["entity1_name"] = entity1["name"]
                new_relationship["entity2_name"] = entity2["name"]

                # 保留兼容性的source/target名称
                if not using_new_format:
                    new_relationship["source_name"] = entity1["name"]
                    new_relationship["target_name"] = entity2["name"]

                # 检查是否重复关系，使用无向的实体对
                # 对实体ID进行排序，确保无向关系的唯一性

                # 排除自环关系（entity1和entity2相同）
                if new_entity1_id == new_entity2_id:
                    logger.warning(f"跳过自环关系: {entity1['name']} -> {entity2['name']}")
                    continue

                entity_pair = tuple(sorted([new_entity1_id, new_entity2_id]))
                rel_key = (entity_pair[0], entity_pair[1], rel_type)

                if rel_key not in relationships_processed:
                    renumbered_relationships.append(new_relationship)
                    relationships_processed.add(rel_key)
            else:
                logger.warning(f"找不到关系对应的实体: {new_entity1_id} - {new_entity2_id}")

        # 为关系重新编号
        for i, relationship in enumerate(renumbered_relationships):
            relationship["id"] = f"r{i+1:04d}"  # 格式：r0001, r0002, ...

        # 记录关系处理结果
        logger.info(f"关系处理结果: 处理前 {len(all_relationships)} 个关系，处理后 {len(renumbered_relationships)} 个关系")

        # 记录前5个关系的源和目标
        if renumbered_relationships:
            logger.info("前5个关系示例:")
            for i, rel in enumerate(renumbered_relationships[:5]):
                logger.info(f"关系 {i+1}: {rel.get('source_name', '')} -> {rel.get('target_name', '')} ({rel.get('type', 'related_to')})")
        else:
            logger.info("警告: 处理后关系列表为空!")
            # 分析原因
            source_name_count = sum(1 for rel in all_relationships if rel.get("source_name"))
            target_name_count = sum(1 for rel in all_relationships if rel.get("target_name"))
            logger.info(f"原始关系中有source_name的数量: {source_name_count}/{len(all_relationships)}")
            logger.info(f"原始关系中有target_name的数量: {target_name_count}/{len(all_relationships)}")
            if all_relationships:
                sample_rel = all_relationships[0]
                logger.info(f"关系样例: source_id={sample_rel.get('source')}, source_name={sample_rel.get('source_name')}, target_id={sample_rel.get('target')}, target_name={sample_rel.get('target_name')}")

        # 更新all_entities和all_relationships，用于后续步骤
        all_entities = renumbered_entities
        all_relationships = renumbered_relationships

        # Save combined matching results
        combined_matching_data = {
            "entities": all_entities,
            "relationships": all_relationships
        }

        with open(combined_matching_path, 'w', encoding='utf-8') as f:
            json.dump(combined_matching_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Combined matching data saved to {combined_matching_path}")

        # 更新匹配统计信息
        self.stats["entities_matched"] = len(all_entities)
        self.stats["relationships_matched"] = len(all_relationships)

        # Ensure we're tracking entities
        logger.info(f"Matching阶段完成：{len(all_entities)} 个实体和 {len(all_relationships)} 个关系")

        # 检查实体ID是否连续从e0001开始
        entity_ids = [entity["id"] for entity in all_entities]
        logger.info(f"实体ID样例: {entity_ids[:5]} ... {entity_ids[-5:] if len(entity_ids) >= 5 else entity_ids}")

        rel_ids = [rel["id"] for rel in all_relationships]
        logger.info(f"关系ID样例: {rel_ids[:5]} ... {rel_ids[-5:] if len(rel_ids) >= 5 else rel_ids}")

        # If no entities but we have relationships, try to recover entities from relationships
        if len(all_entities) == 0 and len(all_relationships) > 0:
            logger.warning("No entities found but relationships exist. Attempting to recover entities from relationships.")
            recovered_entities = {}

            # Process each relationship to extract entities
            for rel in all_relationships:
                source_id = rel.get("source")
                target_id = rel.get("target")
                source_name = rel.get("source_name")
                target_name = rel.get("target_name")

                if source_id and source_name and source_id not in recovered_entities:
                    recovered_entities[source_id] = {
                        "id": source_id,
                        "name": source_name,
                        "type": "concept",
                        "verified": True,
                        "chapter": rel.get("chapter", "")
                    }

                if target_id and target_name and target_id not in recovered_entities:
                    recovered_entities[target_id] = {
                        "id": target_id,
                        "name": target_name,
                        "type": "concept",
                        "verified": True,
                        "chapter": rel.get("chapter", "")
                    }

            all_entities = list(recovered_entities.values())
            logger.info(f"Recovered {len(all_entities)} entities from relationships")

        # Step 3: Entity definition
        # 加载combined_matching.json作为definition步骤的输入
        if not skip_definition:
            logger.info("进入定义模块处理，使用匹配阶段的输出...")

            # 如果已经有combined_matching.json并且要跳过前面的步骤，则从文件中加载
            if skip_ocr or skip_preprocessing or skip_extraction or skip_matching:
                if os.path.exists(combined_matching_path):
                    logger.info(f"从文件加载匹配阶段输出: {combined_matching_path}")
                    try:
                        with open(combined_matching_path, 'r', encoding='utf-8') as f:
                            combined_matching_data = json.load(f)
                        all_entities = combined_matching_data.get("entities", [])
                        all_relationships = combined_matching_data.get("relationships", [])
                        logger.info(f"成功加载匹配阶段输出: {len(all_entities)}个实体，{len(all_relationships)}个关系")
                    except Exception as e:
                        logger.error(f"加载匹配阶段输出出错: {str(e)}")

            # 删除实体处理数量限制
            # 处理所有实体

            # Process all entities
            defined_entities = []
            for entity in all_entities:
                # Define entity if definition is missing
                if "definition" not in entity or not entity["definition"]:
                    # 传递整个entity对象，而不是单独的name和type参数
                    # 获取完整的定义信息，包含definition和significance
                    result = self.entity_definer._get_entity_definition_with_significance(entity)
                    entity["definition"] = result["definition"]
                    entity["significance"] = result["significance"]
                    logger.info(f"为实体 '{entity['name']}' 添加定义和重要性: {result['definition'][:50]}... (重要性: {result['significance']})")
                # 如果有定义但没有significance，添加significance
                elif "significance" not in entity:
                    significance = self.entity_definer._get_entity_significance(entity)
                    entity["significance"] = significance
                    logger.info(f"为实体 '{entity['name']}' 添加重要性: {significance}")
                defined_entities.append(entity)
                logger.info(f"Defined {len(defined_entities)}/{len(all_entities)} entities")

            # Create entity lookup
            entity_dict = {entity["id"]: entity for entity in defined_entities}

            # Process all relationships (删除数量限制)
            defined_relationships = []
            chunk_size = 10  # 每次处理10个关系

            import re

            # 处理完成后，从定义中提取强度值
            def extract_strength_from_definition(definition):
                """从定义文本中提取关系强度"""
                if not definition:
                    return None

                # 尝试匹配格式1：Strength: N 或 strength: N
                match = re.search(r'[Ss]trength:\s*(\d+)', definition)
                if match:
                    try:
                        strength = int(match.group(1))
                        if 1 <= strength <= 3:
                            return strength
                    except ValueError:
                        pass

                # 尝试匹配格式2：纯数字在定义末尾
                match = re.search(r'"?strength"?:\s*(\d+)', definition)
                if match:
                    try:
                        strength = int(match.group(1))
                        if 1 <= strength <= 3:
                            return strength
                    except ValueError:
                        pass

                return None

            logger.info("开始处理关系定义...")
            # Process relationships in chunks
            for i in range(0, len(all_relationships), chunk_size):
                rel_chunk = all_relationships[i:i+chunk_size]
                for rel in rel_chunk:
                    if "definition" not in rel or not rel["definition"]:
                        # 检查关系是否使用新格式(entity1/entity2)或旧格式(source/target)
                        using_new_format = "entity1" in rel and "entity2" in rel

                        if using_new_format:
                            entity1 = entity_dict.get(rel["entity1"])
                            entity2 = entity_dict.get(rel["entity2"])

                            if entity1 and entity2:
                                # 传递整个entity对象和关系类型
                                definition_data = self.entity_definer._get_relationship_definition(
                                    entity1,
                                    entity2,
                                    rel["type"]
                                )
                                # 直接从返回的字典中获取定义和强度
                                rel["definition"] = definition_data.get("definition", "Relationship between the entities.")
                                rel["strength"] = definition_data.get("strength", 0.5)
                        else:
                            # 旧格式，使用source/target
                            source_entity = entity_dict.get(rel["source"])
                            target_entity = entity_dict.get(rel["target"])

                            if source_entity and target_entity:
                                # 传递整个entity对象和关系类型
                                definition_data = self.entity_definer._get_relationship_definition(
                                    source_entity,
                                    target_entity,
                                    rel["type"]
                                )
                                # 直接从返回的字典中获取定义和强度
                                rel["definition"] = definition_data.get("definition", "Relationship between the entities.")
                                rel["strength"] = definition_data.get("strength", 0.5)

                defined_relationships.extend(rel_chunk)
                logger.info(f"Defined {len(defined_relationships)}/{len(all_relationships)} relationships")

            # 处理完所有关系后，检查并从定义中提取强度值
            logger.info("检查关系定义中的强度值...")
            for rel in defined_relationships:
                if rel.get("strength", 1) == 1 and "definition" in rel and rel["definition"]:
                    extracted_strength = extract_strength_from_definition(rel["definition"])
                    if extracted_strength:
                        rel["strength"] = extracted_strength

                        # 根据使用的格式记录日志
                        if "entity1_name" in rel and "entity2_name" in rel:
                            logger.info(f"从定义中提取强度值 {extracted_strength} 用于关系: {rel.get('entity1_name', '')} 和 {rel.get('entity2_name', '')}")
                        else:
                            logger.info(f"从定义中提取强度值 {extracted_strength} 用于关系: {rel.get('source_name', '')} -> {rel.get('target_name', '')}")

            # Save combined definition results
            combined_definition_data = {
                "entities": defined_entities,
                "relationships": defined_relationships
            }

            with open(combined_definition_path, 'w', encoding='utf-8') as f:
                json.dump(combined_definition_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Combined definition data saved to {combined_definition_path}")

            # Update stats
            self.stats["entities_defined"] = len(defined_entities)
            self.stats["relationships_defined"] = len(defined_relationships)

            # Update all_entities and all_relationships for next steps
            all_entities = defined_entities
            all_relationships = defined_relationships
        else:
            # 如果跳过定义步骤，但存在定义输出，则从文件中加载
            if os.path.exists(combined_definition_path):
                logger.info(f"跳过定义步骤，从文件加载定义输出: {combined_definition_path}")
                try:
                    with open(combined_definition_path, 'r', encoding='utf-8') as f:
                        definition_data = json.load(f)
                    all_entities = definition_data.get("entities", all_entities)
                    all_relationships = definition_data.get("relationships", all_relationships)
                    logger.info(f"成功加载定义输出: {len(all_entities)}个实体，{len(all_relationships)}个关系")

                    # 更新统计信息
                    self.stats["entities_defined"] = len(all_entities)
                    self.stats["relationships_defined"] = len(all_relationships)
                except Exception as e:
                    logger.error(f"加载定义输出出错: {str(e)}")

        # Step 4: Canonicalization
        canonicalized_data_path = os.path.join(config.CANONICALIZATION_OUTPUT_DIR, "combined_canonicalized.json")
        if not skip_canonicalization:
            logger.info("进入规范化模块，使用定义阶段的输出...")

            # 检查是否需要从前一阶段文件加载
            if skip_definition:
                # 如果前一阶段被跳过但存在输出，从文件中加载
                if os.path.exists(combined_definition_path):
                    logger.info(f"从文件加载定义阶段输出: {combined_definition_path}")
                    try:
                        with open(combined_definition_path, 'r', encoding='utf-8') as f:
                            definition_data = json.load(f)
                        all_entities = definition_data.get("entities", all_entities)
                        all_relationships = definition_data.get("relationships", all_relationships)
                        logger.info(f"成功加载定义阶段输出: {len(all_entities)}个实体，{len(all_relationships)}个关系")
                    except Exception as e:
                        logger.error(f"加载定义阶段输出出错: {str(e)}")

            # 初始化规范化器
            canonicalizer = RelationshipCanonicalizer(model_name=self.model_name, api_key=self.api_key, api_base=self.api_base)

            # 检查关系中有多少带有定义
            rels_with_def = sum(1 for rel in all_relationships if rel.get("definition", ""))
            logger.info(f"进入规范化前，关系总数: {len(all_relationships)}, 其中有定义的关系数: {rels_with_def}")

            # 如果没有关系带有定义，但我们已经执行了定义步骤，这是一个警告
            if not skip_definition and rels_with_def == 0 and all_relationships:
                logger.warning("虽然执行了定义步骤，但所有关系都没有定义。可能存在数据处理问题。")

            # Process the entities and relationships using the available methods in RelationshipCanonicalizer
            canonicalization_data = {
                "filename": "combined_data",  # Add a filename to avoid the KeyError
                "entities": all_entities,
                "relationships": all_relationships
            }

            try:
                # Use the canonicalize method which handles both entities and relationships
                canonicalized_data = canonicalizer.canonicalize(canonicalization_data)

                # 保存规范化后的数据
                with open(canonicalized_data_path, 'w', encoding='utf-8') as f:
                    json.dump(canonicalized_data, f, indent=2, ensure_ascii=False)
                logger.info(f"规范化数据已保存到 {canonicalized_data_path}")

                # Update entities and relationships with canonicalized versions
                all_entities = canonicalized_data.get("entities", all_entities)
                all_relationships = canonicalized_data.get("relationships", all_relationships)

                # 规范化后检查关系中有多少带有定义
                rels_with_def_after = sum(1 for rel in all_relationships if rel.get("definition", ""))
                logger.info(f"规范化后，关系总数: {len(all_relationships)}, 其中有定义的关系数: {rels_with_def_after}")

                # 如果规范化丢失了定义，记录警告
                if rels_with_def > rels_with_def_after:
                    logger.warning(f"规范化过程丢失了 {rels_with_def - rels_with_def_after} 个关系定义")

                # Save entity types
                entity_types = []
                entity_type_names = set()
                for entity in all_entities:
                    type_field = "canonical_type" if "canonical_type" in entity else "type"
                    entity_type = entity.get(type_field)
                    if entity_type and entity_type not in entity_type_names:
                        entity_types.append({
                            "name": entity_type,
                            "description": f"Entities of type {entity_type}"
                        })
                        entity_type_names.add(entity_type)

                with open(entity_types_path, 'w', encoding='utf-8') as f:
                    json.dump(entity_types, f, indent=2, ensure_ascii=False)
                logger.info(f"Entity types saved to {entity_types_path}")

                # Save relationship types
                relationship_types = []
                relationship_type_names = set()
                for rel in all_relationships:
                    type_field = "canonical_type" if "canonical_type" in rel else "type"
                    rel_type = rel.get(type_field)
                    if rel_type and rel_type not in relationship_type_names:
                        relationship_types.append({
                            "name": rel_type,
                            "description": f"Relationships of type {rel_type}"
                        })
                        relationship_type_names.add(rel_type)

                with open(relationship_types_path, 'w', encoding='utf-8') as f:
                    json.dump(relationship_types, f, indent=2, ensure_ascii=False)
                logger.info(f"Relationship types saved to {relationship_types_path}")

            except Exception as e:
                logger.error(f"Error during canonicalization: {str(e)}")
                logger.error("Continuing with original entities and relationships")
                # Log the traceback for debugging
                import traceback
                logger.error(traceback.format_exc())
        else:
            # 如果跳过规范化步骤，但存在规范化输出，则从文件中加载
            if os.path.exists(canonicalized_data_path):
                logger.info(f"跳过规范化步骤，从文件加载规范化输出: {canonicalized_data_path}")
                try:
                    with open(canonicalized_data_path, 'r', encoding='utf-8') as f:
                        canonicalized_data = json.load(f)
                    all_entities = canonicalized_data.get("entities", all_entities)
                    all_relationships = canonicalized_data.get("relationships", all_relationships)
                    logger.info(f"成功加载规范化输出: {len(all_entities)}个实体，{len(all_relationships)}个关系")
                except Exception as e:
                    logger.error(f"加载规范化输出出错: {str(e)}")

        # Step 5: Exploration
        exploration_data_path = os.path.join(config.EXPLORATION_OUTPUT_DIR, "combined_exploration.json")
        if not skip_exploration:
            logger.info("进入探索模块，使用规范化阶段的输出...")

            # 检查是否需要从前一阶段文件加载
            if skip_canonicalization:
                # 如果前一阶段被跳过但存在输出，从文件中加载
                if os.path.exists(canonicalized_data_path):
                    logger.info(f"从文件加载规范化阶段输出: {canonicalized_data_path}")
                    try:
                        with open(canonicalized_data_path, 'r', encoding='utf-8') as f:
                            canonicalized_data = json.load(f)
                        all_entities = canonicalized_data.get("entities", all_entities)
                        all_relationships = canonicalized_data.get("relationships", all_relationships)
                        logger.info(f"成功加载规范化阶段输出: {len(all_entities)}个实体，{len(all_relationships)}个关系")
                    except Exception as e:
                        logger.error(f"加载规范化阶段输出出错: {str(e)}")

            try:
                # 记录探索前的关系类型分布
                pre_exploration_types = {}
                for rel in all_relationships:
                    rel_type = rel.get("type", "related_to")
                    pre_exploration_types[rel_type] = pre_exploration_types.get(rel_type, 0) + 1
                logger.info(f"Relationship type distribution before exploration: {pre_exploration_types}")

                # Create exploration input in the expected format
                exploration_input = {
                    "filename": "combined_data",
                    "entities": all_entities,
                    "relationships": all_relationships
                }

                # Use the explore method which processes both entities and relationships
                exploration_result = self.type_explorer.explore(exploration_input)

                # 保存探索结果
                with open(exploration_data_path, 'w', encoding='utf-8') as f:
                    json.dump(exploration_result, f, indent=2, ensure_ascii=False)
                logger.info(f"探索结果已保存到 {exploration_data_path}")

                # Extract entity types and relationship types
                entity_types = exploration_result.get("new_entity_types", [])
                relationship_types = exploration_result.get("new_relationship_types", [])

                logger.info(f"Exploration discovered {len(entity_types)} new entity types and {len(relationship_types)} new relationship types")

                # Update entities and relationships if modified during exploration
                if "entities" in exploration_result:
                    all_entities = exploration_result["entities"]
                if "relationships" in exploration_result:
                    all_relationships = exploration_result["relationships"]

                # 记录探索后的关系类型分布
                post_exploration_types = {}
                for rel in all_relationships:
                    rel_type = rel.get("type", "related_to")
                    post_exploration_types[rel_type] = post_exploration_types.get(rel_type, 0) + 1
                logger.info(f"Relationship type distribution after exploration: {post_exploration_types}")

                # 将探索结果的类型信息记录到关系的explored_type字段
                for rel in all_relationships:
                    if rel.get("type") != "related_to":
                        rel["explored_type"] = rel.get("type")

                # Save entity types
                with open(entity_types_path, 'w', encoding='utf-8') as f:
                    json.dump(entity_types, f, indent=2, ensure_ascii=False)
                logger.info(f"Entity types saved to {entity_types_path}")

                # Save relationship types
                with open(relationship_types_path, 'w', encoding='utf-8') as f:
                    json.dump(relationship_types, f, indent=2, ensure_ascii=False)
                logger.info(f"Relationship types saved to {relationship_types_path}")

            except Exception as e:
                logger.error(f"Error during exploration: {str(e)}")
                logger.error("Continuing with original entities and relationships")
                # Log the traceback for debugging
                import traceback
                logger.error(traceback.format_exc())
        else:
            # 如果跳过探索步骤，但存在探索输出，则从文件中加载
            if os.path.exists(exploration_data_path):
                logger.info(f"跳过探索步骤，从文件加载探索输出: {exploration_data_path}")
                try:
                    with open(exploration_data_path, 'r', encoding='utf-8') as f:
                        exploration_result = json.load(f)
                    if "entities" in exploration_result:
                        all_entities = exploration_result["entities"]
                    if "relationships" in exploration_result:
                        all_relationships = exploration_result["relationships"]
                    logger.info(f"成功加载探索输出: {len(all_entities)}个实体，{len(all_relationships)}个关系")
                except Exception as e:
                    logger.error(f"加载探索输出出错: {str(e)}")

        # Step 6: Build knowledge graph
        logger.info("Building knowledge graph...")
        knowledge_graph = self.kg_builder.build(all_entities, all_relationships)

        # Add basic analysis to the knowledge graph
        knowledge_graph["analysis"] = {
            "entity_count": len(all_entities),
            "relationship_count": len(all_relationships),
            "stats": self.stats
        }

        # Extract clean entities and relationships from the knowledge graph
        clean_entities = knowledge_graph["entities"]
        clean_relationships = knowledge_graph["relationships"]

        # Step 7: Update course database
        logger.info("Updating course database...")

        # Create entity ID to entity mapping
        entity_map = {entity["id"]: entity for entity in clean_entities}

        # Get list of processed files
        processed_files = [os.path.basename(pdf_path) for pdf_path in pdf_files]

        # Merge entities and relationships with existing ones in the database
        merged_db_entities = self.db_manager.merge_entities(clean_entities, processed_files)
        merged_db_relationships = self.db_manager.merge_relationships(clean_relationships, entity_map, processed_files)

        # Update entity types and relationship types
        entity_types = []
        entity_type_names = set()
        for entity in merged_db_entities:
            entity_type = entity.get("canonical_type", entity.get("type", "concept"))
            if entity_type not in entity_type_names:
                entity_types.append({
                    "name": entity_type,
                    "description": f"Entities of type {entity_type}"
                })
                entity_type_names.add(entity_type)

        # 创建一个字典，记录各种类型的关系数量
        relationship_type_counts = {}

        relationship_types = []
        relationship_type_names = set()
        for rel in merged_db_relationships:
            # 优先使用explored_type（来自探索阶段），其次是canonical_type（来自规范化阶段），最后才是原始type
            rel_type = rel.get("explored_type", rel.get("canonical_type", rel.get("type", "related_to")))

            # 更新关系类型计数
            relationship_type_counts[rel_type] = relationship_type_counts.get(rel_type, 0) + 1

            if rel_type not in relationship_type_names:
                relationship_types.append({
                    "name": rel_type,
                    "description": f"Relationships of type {rel_type}"
                })
                relationship_type_names.add(rel_type)

        # 记录关系类型分布
        logger.info(f"Final relationship type distribution: {relationship_type_counts}")

        # 保存实体和关系类型
        self.db_manager.save_entity_types(entity_types)
        self.db_manager.save_relationship_types(relationship_types)

        # 保存完整知识图到combined_graph.json
        with open(combined_graph_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_graph, f, indent=2, ensure_ascii=False)
        logger.info(f"Knowledge graph saved to {combined_graph_path}")

        # 从combined_graph.json创建实体和关系文件
        self.db_manager.create_from_combined_graph(combined_graph_path)
        logger.info(f"Created entity and relationship files from combined graph")

        # Step 8: Quality assessment
        if not skip_quality:
            logger.info("Assessing quality of the knowledge graph...")

            # Use the merged database entities and relationships for quality assessment
            merged_knowledge_graph = knowledge_graph.copy()
            merged_knowledge_graph["entities"] = merged_db_entities
            merged_knowledge_graph["relationships"] = merged_db_relationships

            quality_assessment = self.quality_assessor.assess(merged_knowledge_graph)

            # Save quality assessment
            with open(quality_assessment_path, 'w', encoding='utf-8') as f:
                json.dump(quality_assessment, f, indent=2, ensure_ascii=False)
            logger.info(f"Quality assessment saved to {quality_assessment_path}")

        # Mark files as processed
        for pdf_path in pdf_files:
            self.db_manager.mark_file_as_processed(os.path.basename(pdf_path))

        logger.info(f"Pipeline completed. Processed {self.stats['pdfs_processed']} PDFs with {len(merged_db_entities)} entities and {len(merged_db_relationships)} relationships.")
