import streamlit as st
from document_processor import DocumentProcessor
from simple_rag import SimpleRAG
import os
import tempfile



# Paths and files
input_folder = "law"
output_folder = "law/output"
prompts_file = "prompt.txt"

# Load the RAG system
@st.cache_resource
def get_rag_system():
    return SimpleRAG(output_folder, prompts_file)

rag_system = get_rag_system()

# Streamlit interface
st.title("Document Processing and Chatbot Q&A System")

# Process PDF Files section
st.header("Process PDF Files")
if st.button("Process PDFs", key="process_pdfs_button"):
    with st.spinner('Processing PDFs...'):
        try:
            DocumentProcessor.process_folder(input_folder, output_folder)
            st.success("PDF processing complete. Files saved to output folder.")
        except Exception as e:
            st.error(f"An error occurred during PDF processing: {str(e)}")

# Chat with the AI section
st.header("Chat with the AI")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            for response in rag_system.answer_question_stream(prompt):
                full_response += response
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"An error occurred while getting the answer: {str(e)}")
        else:
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# Upload a PDF section
st.header("Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    if st.button("Process Uploaded PDF", key="process_uploaded_pdf_button"):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            DocumentProcessor.process_pdf(tmp_file_path, output_folder)
            st.success("File uploaded and processed successfully!")
        except Exception as e:
            st.error(f"An error occurred while processing the uploaded file: {str(e)}")
        finally:
            if 'tmp_file_path' in locals():
                os.unlink(tmp_file_path)