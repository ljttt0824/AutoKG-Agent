"""Configuration for the subject knowledge graph pipeline."""

import logging
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.absolute()
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
SLIDES_DIR = os.path.join(DATA_DIR, "slides")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

OCR_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "ocr")
PREPROCESSED_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "preprocessed")
EXTRACTION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "extraction")
MATCHING_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "matching")
DEFINITION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "definition")
CANONICALIZATION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "canonicalization")
EXPLORATION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "exploration")

for directory in (
    DATA_DIR,
    OUTPUT_DIR,
    SLIDES_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    OCR_OUTPUT_DIR,
    PREPROCESSED_OUTPUT_DIR,
    EXTRACTION_OUTPUT_DIR,
    MATCHING_OUTPUT_DIR,
    DEFINITION_OUTPUT_DIR,
    CANONICALIZATION_OUTPUT_DIR,
    EXPLORATION_OUTPUT_DIR,
):
    os.makedirs(directory, exist_ok=True)

OCR_DPI = 300
OCR_LANG = os.getenv("OCR_LANG", "eng")
TESSERACT_PATH = os.getenv("TESSERACT_PATH", "")
POPPLER_PATH = os.getenv("POPPLER_PATH", "") or None

DEEPSEEK_MODEL = "deepseek-ai/deepseek-r1-chat"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_MODEL = OPENAI_MODEL

ENTITY_TYPES = ["concept", "protocol", "physical_media", "technology", "other"]
RELATIONSHIP_TYPES = [
    "is_a",
    "part_of",
    "depends_on",
    "equivalent_to",
    "prerequisite_for",
    "sibling_of",
    "related_to",
]

NUM_FEW_SHOT_EXAMPLES = 3
MAX_TOKENS = 1500
TEMPERATURE = 0.2
TOP_P = 0.95
SIMILARITY_THRESHOLD = 0.85
RELATIONSHIP_SIMILARITY_THRESHOLD = 0.75

QUALITY_METRICS = [
    "entity_coverage",
    "relationship_coverage",
    "entity_accuracy",
    "relationship_accuracy",
    "graph_connectivity",
    "graph_density",
]

LOG_LEVEL = logging.INFO
