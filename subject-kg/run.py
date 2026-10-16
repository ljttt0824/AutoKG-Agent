#!/usr/bin/env python
"""
Main script for running the Computer Network Course Knowledge Graph Construction pipeline.
"""

import os
import argparse
import logging
from pathlib import Path

# Import the RunPipeline class
from run_pipeline import RunPipeline

# Import course database manager for checking processed files
from course_db.db_manager import CourseDBManager

# Import configuration
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kg_construction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Computer Network Course Knowledge Graph Construction")

    parser.add_argument("--input_dir", type=str, default=config.SLIDES_DIR,
                        help="Directory containing PDF slides")
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR,
                        help="Directory to save output files")

    # File selection options
    parser.add_argument("--input_file", type=str, default=None,
                        help="Specific PDF file to process (without extension)")
    parser.add_argument("--all_files", action="store_true",
                        help="Process all PDF files in input directory")

    # LLM model settings
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL,
                        help="LLM model to use for extraction and processing")
    # OCR settings
    parser.add_argument("--ocr_dpi", type=int, default=config.OCR_DPI,
                        help="DPI for OCR processing")
    parser.add_argument("--tesseract_path", type=str, default=config.TESSERACT_PATH,
                        help="Path to Tesseract executable")
    parser.add_argument("--poppler_path", type=str, default=config.POPPLER_PATH,
                        help="Path to Poppler bin directory")

    # Skip options
    parser.add_argument("--skip_ocr", action="store_true",
                        help="Skip OCR processing if already done")
    parser.add_argument("--skip_preprocessing", action="store_true",
                        help="Skip preprocessing if already done")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Skip extraction if already done")
    parser.add_argument("--skip_matching", action="store_true",
                        help="Skip matching if already done")
    parser.add_argument("--skip_definition", action="store_true",
                        help="Skip definition if already done")
    parser.add_argument("--skip_canonicalization", action="store_true",
                        help="Skip canonicalization if already done")
    parser.add_argument("--skip_exploration", action="store_true",
                        help="Skip exploration if already done")
    parser.add_argument("--skip_quality", action="store_true",
                        help="Skip quality assessment if already done")
    parser.add_argument("--use_stable_matcher", action="store_true",
                        help="Use stable version of matcher for quality assessment")

    return parser.parse_args()

def main():
    """
    Main function for running the Computer Network Course Knowledge Graph Construction pipeline.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Initialize pipeline
    pipeline = RunPipeline(
        model_name=args.model,
        api_key=config.OPENAI_API_KEY,
        api_base=config.OPENAI_API_BASE,
        ocr_dpi=args.ocr_dpi,
        poppler_path=args.poppler_path,
        tesseract_path=args.tesseract_path,
        output_dir=args.output_dir
    )

    # Get PDF files to process
    pdf_files = []

    if args.input_file:
        # Process a specific PDF
        pdf_path = Path(args.input_dir) / f"{args.input_file}.pdf"
        if not pdf_path.exists():
            logger.error(f"PDF file does not exist: {pdf_path}")
            return
        pdf_files.append(str(pdf_path))
    elif args.all_files:
        # Process all PDF files in the input directory
        for filename in os.listdir(config.SLIDES_DIR):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(config.SLIDES_DIR, filename)
                pdf_files.append(pdf_path)
    else:
        # Find new PDF files to process
        processed_files = pipeline.db_manager.get_processed_files()
        for filename in os.listdir(config.SLIDES_DIR):
            if filename.endswith('.pdf') and filename not in processed_files:
                pdf_path = os.path.join(config.SLIDES_DIR, filename)
                pdf_files.append(pdf_path)

    # Check if there are PDF files to process
    if not pdf_files:
        logger.warning(f"No new PDF files found to process in directory: {config.SLIDES_DIR}")
        logger.warning(f"To process already processed files, use the --all_files flag.")
        if args.skip_quality:
            logger.info("Skipping PDF processing and continuing with quality assessment...")
        else:
            return

    # Log the number of PDF files found
    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")

    # Process all files using the pipeline's run method
    pipeline.run(
        pdf_files=pdf_files,
        skip_ocr=args.skip_ocr,
        skip_preprocessing=args.skip_preprocessing,
        skip_extraction=args.skip_extraction,
        skip_matching=args.skip_matching,
        skip_definition=args.skip_definition,
        skip_canonicalization=args.skip_canonicalization,
        skip_exploration=args.skip_exploration,
        skip_quality=args.skip_quality
    )

if __name__ == "__main__":
    main()
