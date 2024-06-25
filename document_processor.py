import os
import json
from typing import List
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    @staticmethod
    def process_pdf(pdf_path: str, output_folder: str) -> None:
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in tqdm(reader.pages, desc=f"Processing {pdf_path}"):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    images = page.images
                    for image in images:
                        text += pytesseract.image_to_string(Image.open(image)) + "\n"

            chunks = DocumentProcessor.chunk_text(text)
            processed_doc = {
                'source': os.path.basename(pdf_path),
                'chunks': chunks
            }

            output_filename = os.path.splitext(os.path.basename(pdf_path))[0] + '.json'
            with open(os.path.join(output_folder, output_filename), 'w', encoding='utf-8') as f:
                json.dump(processed_doc, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    @staticmethod
    def process_folder(input_folder: str, output_folder: str):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]

        for filename in tqdm(pdf_files, desc="Processing PDF files"):
            pdf_path = os.path.join(input_folder, filename)
            try:
                DocumentProcessor.process_pdf(pdf_path, output_folder)
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
