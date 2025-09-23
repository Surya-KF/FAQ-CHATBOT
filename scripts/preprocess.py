"""
Data preprocessing script for Hospital FAQ documents

This script handles:
- Cleaning and formatting hospital FAQ documents
- Converting various document formats to standardized text
- Removing unnecessary formatting and metadata
- Preparing documents for indexing
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentPreprocessor:
    """Preprocessor for hospital FAQ documents."""
    
    def __init__(self, raw_data_path: str = "./data/raw", processed_data_path: str = "./data/processed"):
        """
        Initialize document preprocessor.
        
        Args:
            raw_data_path: Path to raw documents
            processed_data_path: Path to save processed documents
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        
        # Create directories if they don't exist
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Document preprocessor initialized")
        logger.info(f"Raw data path: {self.raw_data_path}")
        logger.info(f"Processed data path: {self.processed_data_path}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML tags if present
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.,!?;:()\-\'""]', '', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;])', r'\1', text)
        text = re.sub(r'([,.!?;])\s*', r'\1 ', text)
        
        # Remove multiple consecutive punctuation marks
        text = re.sub(r'[,.!?;]{2,}', '.', text)

        # Normalize quotation marks
        text = text.replace('“', '"').replace('”', '"').replace('‟', '"')
        text = text.replace('‘', "'").replace('’', "'").replace('‚', "'")

        # Strip leading/trailing whitespace
        text = text.strip()

        return text
    
    def extract_qa_pairs(self, text: str) -> List[Dict[str, str]]:
        """
        Extract question-answer pairs from text.
        
        Args:
            text: Input text containing Q&A content
            
        Returns:
            List of question-answer dictionaries
        """
        qa_pairs = []
        
        # Pattern to match Q: ... A: ... format
        qa_pattern = r'Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)'
        matches = re.findall(qa_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for question, answer in matches:
            question = self.clean_text(question.strip())
            answer = self.clean_text(answer.strip())
            
            if question and answer:
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "type": "qa_pair"
                })
        
        # If no Q&A pattern found, try other patterns
        if not qa_pairs:
            # Try FAQ: format
            faq_pattern = r'(?:FAQ|Question):\s*(.*?)\s*(?:Answer|Response):\s*(.*?)(?=(?:FAQ|Question):|$)'
            matches = re.findall(faq_pattern, text, re.DOTALL | re.IGNORECASE)
            
            for question, answer in matches:
                question = self.clean_text(question.strip())
                answer = self.clean_text(answer.strip())
                
                if question and answer:
                    qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "type": "faq_pair"
                    })
        
        return qa_pairs
    
    def process_markdown_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Process a Markdown file and extract structured content.
        
        Args:
            file_path: Path to Markdown file
            
        Returns:
            List of processed content blocks
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            processed_blocks = []
            
            # Split by headers
            sections = re.split(r'\n#+\s+', content)
            
            for i, section in enumerate(sections):
                if not section.strip():
                    continue
                
                # Extract header and content
                lines = section.strip().split('\n')
                if i == 0:
                    # First section might not have a header
                    header = "Introduction"
                    section_content = '\n'.join(lines)
                else:
                    header = lines[0] if lines else "Section"
                    section_content = '\n'.join(lines[1:]) if len(lines) > 1 else ""
                
                # Clean the content
                cleaned_content = self.clean_text(section_content)
                
                if cleaned_content:
                    # Try to extract Q&A pairs
                    qa_pairs = self.extract_qa_pairs(cleaned_content)
                    
                    if qa_pairs:
                        processed_blocks.extend(qa_pairs)
                    else:
                        # Add as general content block
                        processed_blocks.append({
                            "title": self.clean_text(header),
                            "content": cleaned_content,
                            "type": "content_block"
                        })
            
            return processed_blocks
            
        except Exception as e:
            logger.error(f"Failed to process markdown file {file_path}: {e}")
            return []
    
    def process_text_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Process a plain text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of processed content blocks
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Clean the content
            cleaned_content = self.clean_text(content)
            
            if not cleaned_content:
                return []
            
            # Try to extract Q&A pairs first
            qa_pairs = self.extract_qa_pairs(cleaned_content)
            
            if qa_pairs:
                return qa_pairs
            
            # If no Q&A pairs, split into paragraphs
            paragraphs = [p.strip() for p in cleaned_content.split('\n\n') if p.strip()]
            
            processed_blocks = []
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) > 50:  # Only include substantial paragraphs
                    processed_blocks.append({
                        "title": f"Section {i+1}",
                        "content": paragraph,
                        "type": "content_block"
                    })
            
            return processed_blocks
            
        except Exception as e:
            logger.error(f"Failed to process text file {file_path}: {e}")
            return []
    
    def process_json_file(self, file_path: Path) -> list:
        """Process a JSON FAQ file with a list of Q&A pairs."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            processed_blocks = []
            if isinstance(data, list):
                for item in data:
                    question = self.clean_text(item.get('question', ''))
                    answer = self.clean_text(item.get('answer', ''))
                    if question and answer:
                        processed_blocks.append({
                            'question': question,
                            'answer': answer,
                            'type': 'qa_pair',
                            'source_file': str(file_path.name),
                            'source_path': str(file_path)
                        })
            return processed_blocks
        except Exception as e:
            logger.error(f"Failed to process JSON file {file_path}: {e}")
            return []

    def process_all_files(self) -> None:
        """Process all files in the raw data directory."""
        if not self.raw_data_path.exists():
            logger.warning(f"Raw data directory does not exist: {self.raw_data_path}")
            return

        # Get all supported files
        supported_extensions = ['.txt', '.md', '.json']
        files_to_process = []

        for ext in supported_extensions:
            files_to_process.extend(self.raw_data_path.glob(f"**/*{ext}"))

        if not files_to_process:
            logger.warning("No files found to process in raw data directory.")
            return

        logger.info(f"Found {len(files_to_process)} files to process")

        all_processed_content = []

        for file_path in files_to_process:
            logger.info(f"Processing file: {file_path}")

            if file_path.suffix.lower() == '.md':
                processed_blocks = self.process_markdown_file(file_path)
            elif file_path.suffix.lower() == '.txt':
                processed_blocks = self.process_text_file(file_path)
            elif file_path.suffix.lower() == '.json':
                processed_blocks = self.process_json_file(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_path}")
                continue

            all_processed_content.extend(processed_blocks)
            logger.info(f"Extracted {len(processed_blocks)} content blocks from {file_path}")

        # Save processed content
        output_file = self.processed_data_path / "processed_faq_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_processed_content, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(all_processed_content)} processed content blocks to {output_file}")

        # Also save as individual text files for easier indexing
        for i, block in enumerate(all_processed_content):
            if block['type'] == 'qa_pair':
                filename = f"qa_pair_{i+1:03d}.txt"
                content = f"Question: {block['question']}\n\nAnswer: {block['answer']}"
            else:
                filename = f"content_block_{i+1:03d}.txt"
                content = f"Title: {block.get('title', 'Untitled')}\n\nContent: {block['content']}"

            individual_file = self.processed_data_path / filename
            with open(individual_file, 'w', encoding='utf-8') as f:
                f.write(content)

        logger.info(f"Saved individual content files to {self.processed_data_path}")


def main():
    """Main function to run document preprocessing."""
    logger.info("Starting hospital FAQ document preprocessing...")
    
    # Initialize preprocessor
    preprocessor = DocumentPreprocessor()
    
    # Process all files
    preprocessor.process_all_files()
    
    logger.info("Document preprocessing completed!")


if __name__ == "__main__":
    main()