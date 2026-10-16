"""
Text preprocessor for cleaning and structuring OCR-extracted text using LLM.
"""

import re
import json
import logging
from typing import Dict, List, Any, Union

import config
from utils.llm_utils import LLMClient

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Class for preprocessing OCR-extracted text from slides using LLM.
    """

    def __init__(self, model_name: str = config.DEFAULT_MODEL, api_key: str = None, api_base: str = None):
        """
        Initialize the text preprocessor.

        Args:
            model_name: Name of the LLM model to use
            api_key: API key for the LLM service
            api_base: API base URL for the LLM service
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base

        # Initialize LLM client for title extraction and text improvement
        self.llm_client = LLMClient(model_name, api_key, api_base)

        logger.info("Initialized text preprocessor with LLM capabilities")

    def process(self, ocr_result: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process OCR results to clean and structure the text using LLM.

        Args:
            ocr_result: JSON string or Dictionary containing OCR results

        Returns:
            Dictionary containing preprocessed text with only page numbers, titles, and LLM-processed content
        """
        # If ocr_result is a string, parse it as JSON
        if isinstance(ocr_result, str):
            try:
                ocr_result = json.loads(ocr_result)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing OCR result JSON: {e}")
                # Return empty result
                return {
                    "filename": "unknown",
                    "chapter": "unknown",
                    "pages": []
                }

        logger.info(f"Preprocessing text from {ocr_result['filename']}")

        simplified_result = {
            "filename": ocr_result["filename"],
            "chapter": ocr_result["chapter"],
            "pages": []
        }

        # Process each page with LLM
        for page_data in ocr_result["pages"]:
            # Extract raw text from all blocks
            texts = [block["text"] for block in page_data["blocks"] if block["text"].strip()]
            raw_text = "\n---BLOCK---\n".join(texts)

            # Organize text with LLM
            organized_text = self._organize_text_with_llm(raw_text, page_data["page_number"])

            # Create page object with minimal information
            page = {
                "page_number": page_data["page_number"],
                "content": organized_text,
                "title": ""  # Will be filled later
            }

            simplified_result["pages"].append(page)

        # Extract titles for all pages
        simplified_result = self._extract_titles(simplified_result)

        logger.info(f"Processed {len(simplified_result['pages'])} pages with LLM")
        return simplified_result

    def _extract_titles(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract slide titles for each page using LLM.

        Args:
            result: Dictionary containing preprocessed data

        Returns:
            Updated result dictionary with titles
        """
        logger.info("Extracting titles for all pages using LLM")

        for page in result["pages"]:
            # Skip pages with no text
            if not page["content"].strip():
                page["title"] = f"Empty Page {page['page_number']}"
                continue

            # Generate a prompt for the LLM
            prompt = self._create_title_extraction_prompt(page["content"], page["page_number"])

            try:
                # Call LLM
                response = self.llm_client.call(prompt, response_format="json")
                title_data = json.loads(response)

                # Extract title
                title = title_data.get("title", "")

                # Update page title
                page["title"] = title

                logger.info(f"Extracted title for page {page['page_number']}: {title}")
            except Exception as e:
                logger.error(f"Error extracting title for page {page['page_number']}: {e}")
                # Fallback to first line as title
                first_line = page["content"].split('\n')[0] if '\n' in page["content"] else page["content"]
                if len(first_line) > 50:
                    first_line = first_line[:47] + "..."
                page["title"] = first_line

        return result

    def _create_title_extraction_prompt(self, text: str, page_number: int) -> str:
        """
        Create a prompt for title extraction.

        Args:
            text: Page text
            page_number: Page number

        Returns:
            Prompt for the LLM
        """
        return f"""You are an expert in analyzing academic slides about computer networks.

TASK: Extract the most likely title of this slide (page {page_number}) and evaluate your confidence.

SLIDE CONTENT:
```
{text[:2000]}  # Limit to 2000 chars to avoid token limits
```

Analyze the text and extract the most likely slide title. Look for the following indicators:
1. Text that appears at the top of the slide
2. Text formatted differently (all caps, larger font, etc.)
3. Short phrases that summarize the slide content
4. Common slide title patterns (like "Introduction to X", "Types of Y", etc.)
5. Section headers or numbered list items that could be titles

Return your analysis as a JSON object with these fields:
- "title": The extracted title (should be concise, focus on the main topic)
- "confidence": A decimal number from 0.0 to 1.0 representing your confidence in this title extraction

Example response:
{{"title": "Introduction to TCP/IP Protocol", "confidence": 0.85}}

IMPORTANT: If you can't find a clear title, use the first significant line or create a brief descriptive title based on the content. In that case, set a lower confidence value.

JSON response:"""

    def _organize_text_with_llm(self, raw_text: str, page_number: int) -> str:
        """
        Use LLM to organize text in a logical reading order and improve readability.

        Args:
            raw_text: Raw text from OCR with block markers
            page_number: Page number for logging

        Returns:
            Organized and cleaned text
        """
        # If no text, return empty string
        if not raw_text.strip():
            return ""

        # Create prompt for text organization
        prompt = self._create_text_organization_prompt(raw_text, page_number)

        try:
            # Call LLM
            organized_text = self.llm_client.call(prompt, response_format="text")
            return organized_text.strip()
        except Exception as e:
            logger.error(f"Error organizing text for page {page_number}: {e}")
            # Fallback to original text with block markers removed
            return raw_text.replace("\n---BLOCK---\n", "\n\n")

    def _create_text_organization_prompt(self, text: str, page_number: int) -> str:
        """
        Create a prompt for text organization.

        Args:
            text: Raw text with block markers
            page_number: Page number

        Returns:
            Prompt for the LLM
        """
        return f"""You are an expert in organizing and cleaning OCR-extracted text from academic slides about computer networks.

TASK: Reorganize this OCR text from page {page_number} into a coherent, logical reading order.

RAW TEXT (separated by block markers):
```
{text[:3000]}  # Limit to 3000 chars to avoid token limits
```

INSTRUCTIONS:
1. EXTREMELY IMPORTANT: Preserve ALL technical terminology, concepts, protocols, and academic terms exactly as they appear
2. DO NOT omit any computer network concepts, terminology, or topics, even if they appear only briefly
3. Preserve all bullets, lists, and hierarchical structures
4. Ensure ALL original content is retained - missing ANY technical term is a serious error
5. Organize text in a logical reading order (usually top-to-bottom, left-to-right)
6. Fix obvious OCR errors in technical terms without changing their meaning
7. Remove duplicate information, page numbers, footers, and irrelevant metadata
8. Preserve all section titles and headings

FORMAT YOUR RESPONSE AS CLEAN TEXT ONLY. Do not add any explanations, just output the reorganized text.

IMPORTANT: Your primary goal is to preserve ALL technical content exactly as it appears. Only organize and clean the text, NEVER remove or simplify technical terms. Be extremely careful not to miss any concepts mentioned in the original text."""

    def batch_pages_for_extraction(self, result: Dict[str, Any], batch_size: int = 3) -> List[Dict[str, Any]]:
        """
        Create batches of pages for entity and relationship extraction.

        Args:
            result: Dictionary containing preprocessed data with titles
            batch_size: Maximum number of pages in a batch

        Returns:
            List of batch dictionaries
        """
        batches = []
        pages = result["pages"]

        # 创建简化格式的批次
        for i in range(0, len(pages), batch_size):
            batch_pages = pages[i:i+batch_size]

            batch = {
                "batch_title": f"Pages {batch_pages[0]['page_number']}-{batch_pages[-1]['page_number']}",
                "page_count": len(batch_pages),
                "page_numbers": [page["page_number"] for page in batch_pages],
                "pages": batch_pages,
                "combined_text": "\n\n".join([f"PAGE {page['page_number']}:\n{page['content']}" for page in batch_pages])
            }

            batches.append(batch)

        logger.info(f"Created {len(batches)} page batches for extraction")
        return batches

    def group_pages_by_title(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        将页面按标题相似性分组用于关系抽取。

        Args:
            result: 包含预处理数据及标题的字典

        Returns:
            按标题分组的页面列表
        """
        logger.info("将页面按标题相似性分组用于关系抽取")

        # 创建标题规范化函数
        def normalize_title(title):
            # 转为小写
            normalized = title.lower()
            # 移除常见前缀（如"Chapter X:"，"Section X:"等）
            normalized = re.sub(r'^(chapter|section|part|unit)\s+\d+[:.]\s*', '', normalized)
            # 移除标点符号和数字
            normalized = re.sub(r'[0-9.,;:!?()[\]{}\'\"]+', '', normalized)
            # 移除不必要的词
            filler_words = ['a', 'an', 'the', 'and', 'or', 'but', 'of', 'to', 'in', 'on', 'for', 'with', 'by', 'about']
            normalized_words = [word for word in normalized.split() if word not in filler_words]
            # 连接剩余的词
            return ' '.join(normalized_words).strip()

        # 按规范化标题分组
        title_groups = {}
        for page in result["pages"]:
            normalized_title = normalize_title(page["title"])
            if normalized_title not in title_groups:
                title_groups[normalized_title] = []
            title_groups[normalized_title].append(page)

        # 创建分组结果
        grouped_pages = []
        for normalized_title, pages in title_groups.items():
            if not normalized_title:  # 跳过空标题
                continue

            group = {
                "group_title": pages[0]["title"],
                "normalized_title": normalized_title,
                "page_count": len(pages),
                "page_numbers": [page["page_number"] for page in pages],
                "pages": pages,
                "combined_content": "\n\n".join([f"PAGE {page['page_number']}:\n{page['content']}" for page in pages])
            }
            grouped_pages.append(group)

        logger.info(f"按标题分组创建了{len(grouped_pages)}个页面组")
        return grouped_pages
