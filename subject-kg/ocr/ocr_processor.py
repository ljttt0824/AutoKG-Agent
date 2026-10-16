"""
OCR processor for extracting text and layout information from PDF slides.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Union

import pytesseract
# 设置Tesseract路径（Windows用户需要）

from pdf2image import convert_from_path
import cv2
import numpy as np
from PIL import Image

import config

logger = logging.getLogger(__name__)

class OCRProcessor:
    """
    Class for extracting text and layout information from PDF slides using OCR.
    """

    def __init__(self, dpi: int = 300, lang: str = "eng", poppler_path: str = None):
        """
        Initialize the OCR processor.

        Args:
            dpi: DPI for PDF to image conversion
            lang: Language for OCR processing
            poppler_path: Path to poppler binaries (for pdf2image)
        """
        self.dpi = dpi
        self.lang = lang
        self.poppler_path = poppler_path
        logger.info(f"Initialized OCR processor with DPI={dpi}, language={lang}")

    def process(self, pdf_path: Union[str, Path]) -> str:
        """
        Process a PDF file and extract text and layout information.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            JSON string containing extracted text and layout information
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Processing PDF: {pdf_path}")

        # Extract chapter information from filename or first page
        chapter_info = self._extract_chapter_info(pdf_path)

        # Convert PDF to images
        try:
            # 使用poppler_path参数
            images = convert_from_path(pdf_path, dpi=self.dpi, poppler_path=self.poppler_path)
            logger.info(f"Converted PDF to {len(images)} images")
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            # 返回空结果
            result = {
                "filename": pdf_path.name,
                "chapter": chapter_info,
                "pages": []
            }
            return json.dumps(result, ensure_ascii=False, indent=2)

        result = {
            "filename": pdf_path.name,
            "chapter": chapter_info,
            "pages": []
        }

        # Process each page
        for i, image in enumerate(images):
            page_num = i + 1
            logger.info(f"Processing page {page_num}/{len(images)}")

            # Convert PIL image to OpenCV format
            img_np = np.array(image)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Extract text with layout information
            page_data = self._process_page(img_cv, page_num)

            # Add page data to result
            result["pages"].append(page_data)

        # Convert result to JSON string
        return json.dumps(result, ensure_ascii=False, indent=2)

    def _extract_chapter_info(self, pdf_path: Path) -> str:
        """
        Extract chapter information from the PDF filename or first page.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Chapter information as a string
        """
        # Try to extract chapter info from filename
        filename = pdf_path.stem
        if "chapter" in filename.lower() or "chapitre" in filename.lower():
            return filename

        # If not in filename, convert first page and try to find chapter info
        try:
            first_page = convert_from_path(pdf_path, dpi=self.dpi, first_page=1, last_page=1, poppler_path=self.poppler_path)[0]
            img_np = np.array(first_page)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # OCR the first page
            text = pytesseract.image_to_string(img_cv, lang=self.lang)

            # Look for chapter information in the first few lines
            lines = text.split('\n')
            for line in lines[:10]:
                if "chapter" in line.lower() or "chapitre" in line.lower():
                    return line.strip()

            # If no chapter info found, use filename
            return filename
        except Exception as e:
            logger.error(f"Error extracting chapter info: {e}")
            return filename

    def _process_page(self, img: np.ndarray, page_num: int) -> Dict[str, Any]:
        """
        Process a single page image and extract text with layout information.

        Args:
            img: OpenCV image of the page
            page_num: Page number

        Returns:
            Dictionary containing page data with text and layout information
        """
        # Get page dimensions
        height, width, _ = img.shape

        # Use pytesseract to get text and bounding boxes
        try:
            ocr_data = pytesseract.image_to_data(img, lang=self.lang, output_type=pytesseract.Output.DICT)
        except Exception as e:
            logger.error(f"Error performing OCR on page {page_num}: {e}")
            # 返回空页面数据
            return {
                "page_number": page_num,
                "width": width,
                "height": height,
                "blocks": [],
                "full_text": ""
            }

        # Extract text blocks with their positions
        blocks = []
        current_block = {"text": "", "bbox": None, "is_title": False, "confidence": 0}

        for i in range(len(ocr_data["text"])):
            text = ocr_data["text"][i].strip()
            conf = int(ocr_data["conf"][i])

            if not text:
                continue

            # Get bounding box
            x = ocr_data["left"][i]
            y = ocr_data["top"][i]
            w = ocr_data["width"][i]
            h = ocr_data["height"][i]

            # Check if this is a new block
            if ocr_data["block_num"][i] != ocr_data.get("block_num", [0])[i-1] and current_block["text"]:
                blocks.append(current_block)
                current_block = {"text": "", "bbox": None, "is_title": False, "confidence": 0}

            # Update current block
            if not current_block["text"]:
                current_block["bbox"] = [x, y, x+w, y+h]
                current_block["confidence"] = conf
            else:
                current_block["text"] += " "
                # Expand bounding box
                current_block["bbox"][2] = max(current_block["bbox"][2], x+w)
                current_block["bbox"][3] = max(current_block["bbox"][3], y+h)
                current_block["confidence"] = (current_block["confidence"] + conf) / 2

            current_block["text"] += text

        # Add the last block if not empty
        if current_block["text"]:
            blocks.append(current_block)

        # Identify titles based on font size, position, etc.
        blocks = self._identify_titles(blocks, height, width)

        return {
            "page_number": page_num,
            "width": width,
            "height": height,
            "blocks": blocks,
            "full_text": " ".join([block["text"] for block in blocks])
        }

    def _identify_titles(self, blocks: List[Dict[str, Any]], height: int, width: int) -> List[Dict[str, Any]]:
        """
        Identify which text blocks are likely titles based on position, size, etc.

        Args:
            blocks: List of text blocks
            height: Page height
            width: Page width

        Returns:
            Updated list of text blocks with is_title flag set
        """
        if not blocks:
            return blocks

        # Sort blocks by vertical position
        blocks_sorted = sorted(blocks, key=lambda b: b["bbox"][1])

        # The first block is often a title
        if blocks_sorted[0]["bbox"][1] < height * 0.2:
            blocks_sorted[0]["is_title"] = True

        # Look for centered blocks that might be titles
        for block in blocks_sorted:
            block_center_x = (block["bbox"][0] + block["bbox"][2]) / 2
            if abs(block_center_x - width/2) < width * 0.1:
                block["is_title"] = True

        return blocks_sorted
