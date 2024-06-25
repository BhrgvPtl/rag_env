from document_processor import DocumentProcessor
from simple_rag import SimpleRAG
import os

# Paths and files
input_folder = "C://Users//bhargav//OneDrive//Desktop//law"
output_folder = "C://Users//bhargav//OneDrive//Desktop//law//output"
prompts_file = "prompt.txt"

def main():
    # Process PDFs if needed
    process_pdfs = input("Do you want to process PDFs? (y/n): ").lower() == 'y'
    if process_pdfs:
        try:
            DocumentProcessor.process_folder(input_folder, output_folder)
            print("PDF processing complete. Files saved to output folder.")
        except Exception as e:
            print(f"An error occurred during PDF processing: {str(e)}")

    # Initialize RAG system
    rag_system = SimpleRAG(output_folder, prompts_file)

    # Chat loop
    while True:
        user_query = input("Enter your question (or 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break

        try:
            answer = rag_system.answer_question_stream(user_query)
            print(f"AI: {answer}")
        except Exception as e:
            print(f"An error occurred while getting the answer: {str(e)}")

if __name__ == "__main__":
    main()