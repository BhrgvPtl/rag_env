import os
import json
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import openai
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

class SimpleRAG:
    def __init__(self, preprocessed_folder: str, prompts_file: str):
        self.preprocessed_folder = preprocessed_folder
        self.documents = self.load_preprocessed_documents()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_vectors = self.compute_chunk_vectors()
        self.index = faiss.IndexFlatL2(self.chunk_vectors.shape[1])
        self.index.add(np.array(self.chunk_vectors).astype('float32'))
        self.prompts = self.load_prompts(prompts_file)

    def load_preprocessed_documents(self) -> List[Dict]:
        documents = []
        json_files = [f for f in os.listdir(self.preprocessed_folder) if f.endswith('.json')]
        logger.info(f"Found {len(json_files)} JSON files in {self.preprocessed_folder}")
        
        for filename in tqdm(json_files, desc="Loading preprocessed documents"):
            try:
                with open(os.path.join(self.preprocessed_folder, filename), 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    documents.append(doc)
                    logger.info(f"Loaded document from {filename} with {len(doc['chunks'])} chunks")
            except Exception as e:
                logger.error(f"Error loading {filename}: {str(e)}")
        
        logger.info(f"Loaded a total of {len(documents)} documents")
        logger.info(f"Total chunks across all documents: {sum(len(doc['chunks']) for doc in documents)}")
        
        return documents
    

    def compute_chunk_vectors(self) -> np.ndarray:
        all_chunks = [chunk for doc in self.documents for chunk in doc['chunks']]
        return self.model.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True)

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        query_vector = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_vector.astype('float32'), top_k)
        
        all_chunks = [chunk for doc in self.documents for chunk in doc['chunks']]
        relevant_chunks = [all_chunks[i] for i in I[0]]
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
        return relevant_chunks
    
    def load_prompts(self, prompts_file: str) -> Dict:
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                content = f.read()
                logging.info(f"Raw content of prompts file:\n{content}")
                prompts = json.loads(content)
            if isinstance(prompts, list):
                logging.info(f"Loaded {len(prompts)} prompts")
                # Convert the list of prompts to a dictionary
                prompts_dict = {
                    "prompts": prompts,
                    "default_prompt": "Based on the following context:\n\n{context}\n\nPlease answer this question: {query}"
                }
                return prompts_dict
            elif isinstance(prompts, dict):
                logging.info(f"Loaded prompts: {prompts}")
                return prompts
            else:
                raise ValueError(f"Prompts file contains {type(prompts)}, expected a list or dictionary")
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding error in prompts file: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error loading prompts: {str(e)}")
            raise
    def answer_question_stream(self, query: str):
        try:
            relevant_chunks = self.retrieve_relevant_chunks(query)
            context = " ".join(relevant_chunks)
            
            logger.info(f"Retrieved context (truncated): {context[:500]}...")
            
            prompt = self.prompts.get("default_prompt", "Based on the following context:\n\n{context}\n\nPlease answer this question: {query}")
            logger.info(f"Raw prompt template: {prompt}")
            
            formatted_prompt = prompt.format(context=context, query=query)
            logger.info(f"Formatted prompt (truncated): {formatted_prompt[:500]}...")
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
                    {"role": "user", "content": formatted_prompt}
                ],
                stream=True
            )
            
            for chunk in response:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    if 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                        yield chunk['choices'][0]['delta']['content']
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}", exc_info=True)
            yield f"An error occurred: {str(e)}"
