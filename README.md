# PDF RAG Chat

This is a simple RAG (Retrieval-Augmented Generation) application that allows you to upload PDF files and chat with their contents using Ollama language models. The app extracts text from PDFs, chunks it, creates embeddings, and uses vector similarity search to find relevant information for answering queries.

## Features

- Upload multiple PDF files
- Chat with your documents using Ollama language models
- Automatic text chunking and embedding
- Retrieval of relevant document sections for accurate answers
- Adjustable model parameters
- Conversation history

## Requirements

- Python 3.8+
- Ollama installed and running on your system
- PDF files you want to chat with

## Installation

1. Clone this repository or download the files

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Make sure Ollama is installed and running on your system.
   - For macOS: Download from [ollama.com/download](https://ollama.com/download) or install via Homebrew with `brew install ollama`
   - Start Ollama with `ollama serve` or by launching the application

## Usage

1. Start the application:

```bash
streamlit run app.py
```

2. Open your web browser and navigate to http://localhost:8501

3. Upload your PDF files using the file uploader in the sidebar

4. Click "Process PDFs" to extract text, create chunks, and generate embeddings

5. Ask questions about your documents in the chat interface

## How It Works

1. **PDF Processing**: Extracts text from uploaded PDF files
2. **Text Chunking**: Splits text into smaller, overlapping chunks for better retrieval
3. **Embedding Generation**: Creates vector embeddings for each chunk using Ollama
4. **Vector Storage**: Stores embeddings in a FAISS index for efficient similarity search
5. **Query Processing**: 
   - Converts user queries to embeddings
   - Finds most similar document chunks
   - Uses found chunks as context for Ollama
   - Generates responses based on the provided context

## Customization

- **Model Selection**: Choose any Ollama model you have downloaded
- **Temperature**: Adjust response creativity (0.0-1.0)
- **System Prompt**: Customize the base instructions for the AI assistant

## Limitations

- PDF text extraction may not work perfectly for all PDF types (especially scanned documents)
- Processing very large PDFs or many PDFs simultaneously may require significant memory
- Response quality depends on the Ollama model used and the quality of text extraction
