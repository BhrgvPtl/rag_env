# Law Document Retrieval and Question Answering System

This project is a Law Document Retrieval and Question Answering System using Retrieval-Augmented Generation (RAG). It processes PDF files to extract and chunk text, embeds these chunks, and allows users to query the documents using a Streamlit interface. The system uses OpenAI's GPT-3.5-turbo to provide answers based on the retrieved document chunks.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Contributing](#contributing)
- [License](#license)

## Features

- Process PDF files to extract and chunk text.
- Embed document chunks using Sentence Transformers.
- Retrieve relevant document chunks based on user queries.
- Generate answers to queries using OpenAI's GPT-3.5-turbo.
- Streamlit-based user interface for processing PDFs and querying documents.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/BhrgvPtl/rag_env.git
    cd rag_env
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Set your OpenAI API key as an environment variable:

    ```bash
    export OPENAI_API_KEY="your-api-key-here"  # On Windows, use `set OPENAI_API_KEY="your-api-key-here"`
    ```

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2. Use the web interface to:
    - Process PDF files in the specified input folder.
    - Upload and process a single PDF file.
    - Query the processed documents and get answers based on the retrieved document chunks.

## File Descriptions

### `app.py`

The main application file that runs the Streamlit interface. It allows users to process PDF files, upload and process individual PDF files, and interact with the chatbot.

### `document_processor.py`

Contains the `DocumentProcessor` class with methods to process PDF files. It extracts text from PDFs, optionally using OCR for images, chunks the text, and saves the processed documents in JSON format.

### `simple_rag.py`

Contains the `SimpleRAG` class that implements the Retrieval-Augmented Generation system. It loads preprocessed documents, embeds the text chunks, retrieves relevant chunks for a given query, and uses OpenAI's GPT-3.5-turbo to generate answers.

### `requirements.txt`

A list of dependencies required to run the project. These can be installed using pip.

### `prompt.txt`

A file containing prompts for the RAG system. It can be customized as needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
