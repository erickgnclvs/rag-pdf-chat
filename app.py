#!/usr/bin/env python3

import os
import tempfile
import numpy as np
import streamlit as st
import PyPDF2
import faiss
import ollama
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="PDF RAG Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title
st.title("📚 PDF RAG Chat")
st.markdown("Upload PDF files and chat with your documents")

# Constants
DEFAULT_MODEL = "tinyllama"
DEFAULT_TEMPERATURE = 0.7
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_SIZE = 384  # Default for Ollama models

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
    
if "document_embeddings" not in st.session_state:
    st.session_state.document_embeddings = None
    
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []
    
if "is_initialized" not in st.session_state:
    st.session_state.is_initialized = False

# Check if Ollama is available
def check_ollama_available():
    try:
        ollama.list()
        return True
    except Exception as e:
        st.error(f"Error connecting to Ollama: {e}")
        st.markdown("""
        ### Ollama is not running or not installed
        
        Please make sure Ollama is installed and running on your system.
        
        - **macOS**: Install from [ollama.com/download](https://ollama.com/download) or via Homebrew with `brew install ollama`
        - Then start Ollama and refresh this page
        """)
        return False

# Get available models from Ollama
def get_available_models():
    try:
        models = ollama.list()
        
        # The most common response format is {'models': [{'name': 'model1'}, {'name': 'model2'}, ...]}
        if isinstance(models, dict) and 'models' in models:
            if isinstance(models['models'], list):
                return [model.get('name', 'unknown') for model in models['models']]
        
        # Some versions might return a direct list
        elif isinstance(models, list):
            return [model.get('name', model.get('model', 'unknown')) for model in models]
            
        # If we can't extract model names, return a default list without showing a warning
        return ["tinyllama", "llama2", "mistral", "gemma:2b", "phi"]
    except Exception as e:
        # Silently handle the error and return fallback models without showing an error message
        return ["tinyllama", "llama2", "mistral", "gemma:2b", "phi"]

# Extract text from PDF files
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_file.getvalue())
            temp_path = temp_file.name
        
        with open(temp_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        
        # Clean up the temporary file
        os.remove(temp_path)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    
    return text

# Split text into chunks
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    
    if not text.strip():
        return chunks
    
    # Split by paragraphs first to maintain some context
    paragraphs = text.split("\n\n")
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If paragraph is larger than chunk_size, split it further
            if len(paragraph) > chunk_size:
                words = paragraph.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= chunk_size:
                        current_chunk += word + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        
                        # Use overlap for next chunk
                        words_in_chunk = current_chunk.split()
                        overlap_words = words_in_chunk[-min(len(words_in_chunk), overlap // 10):]
                        current_chunk = " ".join(overlap_words) + " " + word + " "
            else:
                current_chunk = paragraph + "\n\n"
    
    # Add the last chunk if not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

# Get embeddings from Ollama
def get_embedding(text, model_name):
    try:
        response = ollama.embeddings(model=model_name, prompt=text)
        return response['embedding']
    except Exception as e:
        st.error(f"Error getting embeddings: {e}")
        # Return a zero embedding as fallback
        return [0.0] * EMBEDDING_SIZE

# Create FAISS index for efficient similarity search
def create_faiss_index(embeddings):
    if not embeddings or len(embeddings) == 0:
        return None
    
    # Convert list of embeddings to numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Get dimensions
    vector_dimension = embeddings_array.shape[1]
    
    # Create FAISS index
    index = faiss.IndexFlatL2(vector_dimension)
    index.add(embeddings_array)
    
    return index

# Find most similar chunks to a query
def search_similar_chunks(query_embedding, top_k=3):
    if st.session_state.faiss_index is None or len(st.session_state.document_chunks) == 0:
        return []
    
    # Convert query embedding to numpy array
    query_array = np.array([query_embedding]).astype('float32')
    
    # Search in FAISS index
    distances, indices = st.session_state.faiss_index.search(query_array, min(top_k, len(st.session_state.document_chunks)))
    
    # Return the relevant chunks
    return [(st.session_state.document_chunks[idx], float(distances[0][i])) for i, idx in enumerate(indices[0])]

# Generate context from similar chunks
def generate_context(similar_chunks, max_tokens=1500):
    context = "Here are the most relevant passages from the documents:\n\n"
    
    # Sort by similarity (lowest distance first)
    similar_chunks.sort(key=lambda x: x[1])
    
    for chunk, distance in similar_chunks:
        # Add chunk if there's space
        if len(context) + len(chunk) <= max_tokens:
            context += chunk + "\n\n"
    
    return context

# Generate response from Ollama
def generate_response(query, model_name, temperature, system_prompt):
    # Get query embedding
    query_embedding = get_embedding(query, model_name)
    
    # Get similar chunks
    similar_chunks = search_similar_chunks(query_embedding, top_k=3)
    
    if not similar_chunks:
        return "I don't have enough information in the provided documents to answer your question."
    
    # Generate context from similar chunks
    context = generate_context(similar_chunks)
    
    # Create messages for Ollama
    messages = [
        {
            "role": "system",
            "content": f"{system_prompt}\n\nContent from documents: {context}"
        }
    ]
    
    # Add conversation history
    for message in st.session_state.conversation_history:
        messages.append(message)
    
    # Add user query
    messages.append({"role": "user", "content": query})
    
    # Generate response from Ollama
    try:
        response = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            for chunk in ollama.chat(
                model=model_name,
                messages=messages,
                stream=True,
                options={"temperature": temperature}
            ):
                if 'message' in chunk and 'content' in chunk['message']:
                    content_chunk = chunk['message']['content']
                    response += content_chunk
                    message_placeholder.markdown(response + "▌")
            
            message_placeholder.markdown(response)
        
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I encountered an error while generating a response."

# Process uploaded PDFs
def process_pdfs(uploaded_files):
    if not uploaded_files:
        return
    
    # Get the selected model from session state
    model_name = st.session_state.get('selected_model', DEFAULT_MODEL)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Reset document state
    st.session_state.document_chunks = []
    st.session_state.document_embeddings = []
    st.session_state.pdf_names = []
    
    total_files = len(uploaded_files)
    
    for i, pdf_file in enumerate(uploaded_files):
        # Update progress
        progress = int((i / total_files) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing {pdf_file.name}... ({i+1}/{total_files})")
        
        # Extract text from PDF
        st.session_state.pdf_names.append(pdf_file.name)
        text = extract_text_from_pdf(pdf_file)
        
        if not text.strip():
            st.warning(f"No text could be extracted from {pdf_file.name}")
            continue
        
        # Chunk the text
        chunks = chunk_text(text)
        
        # Get embeddings for each chunk
        for chunk in chunks:
            embedding = get_embedding(chunk, model_name)
            st.session_state.document_chunks.append(chunk)
            st.session_state.document_embeddings.append(embedding)
    
    # Create FAISS index
    if st.session_state.document_embeddings:
        st.session_state.faiss_index = create_faiss_index(st.session_state.document_embeddings)
    
    # Complete progress
    progress_bar.progress(100)
    status_text.text(f"Processed {total_files} PDF files with {len(st.session_state.document_chunks)} chunks")
    
    # Set initialization status
    st.session_state.is_initialized = True

# Reset conversation
def reset_conversation():
    st.session_state.conversation_history = []

# Main sidebar content
def render_sidebar():
    with st.sidebar:
        st.header("Settings")
        
        # Model settings
        st.subheader("Model Configuration")
        
        # Check if Ollama is available and get models
        if check_ollama_available():
            available_models = get_available_models()
            
            # Store model selection in session state
            st.session_state.selected_model = st.selectbox(
                "Select model",
                options=available_models,
                index=available_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in available_models else 0
            )
            
            # Store temperature in session state
            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_TEMPERATURE,
                step=0.1,
                help="Higher values make output more random, lower values more deterministic"
            )
            
            # Store system prompt in session state
            st.session_state.system_prompt = st.text_area(
                "System prompt",
                value="You are a helpful AI assistant that answers questions based on the provided documents. Provide concise, accurate responses based on the document content. If the answer is not in the documents, say so clearly.",
                help="Initial instruction to guide the model's behavior"
            )
        
        # PDF upload
        st.subheader("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type="pdf",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            process_button = st.button("Process PDFs")
            if process_button:
                process_pdfs(uploaded_files)
        
        # Display uploaded PDFs
        if st.session_state.pdf_names:
            st.subheader("Processed Documents")
            for pdf_name in st.session_state.pdf_names:
                st.text(f"• {pdf_name}")
        
        # Reset buttons
        if st.session_state.is_initialized:
            if st.button("Reset Conversation"):
                reset_conversation()
                st.success("Conversation history cleared!")
            
            if st.button("Clear Documents"):
                st.session_state.document_chunks = []
                st.session_state.document_embeddings = []
                st.session_state.faiss_index = None
                st.session_state.pdf_names = []
                st.session_state.is_initialized = False
                st.success("Documents cleared!")

# Main chat interface
def render_chat_interface():
    # Display conversation history
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:  # assistant
            with st.chat_message("assistant"):
                st.markdown(message["content"])
    
    # Chat input
    if query := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.conversation_history.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Check if documents have been processed
        if not st.session_state.is_initialized:
            with st.chat_message("assistant"):
                st.markdown("Please upload and process PDF documents first!")
        else:
            # Get model settings from session state to avoid duplicate widgets
            # These widgets are already created in the sidebar
            # Generate and display assistant response
            response = generate_response(
                query=query,
                model_name=st.session_state.get('selected_model', DEFAULT_MODEL),
                temperature=st.session_state.get('temperature', DEFAULT_TEMPERATURE),
                system_prompt=st.session_state.get('system_prompt', "You are a helpful AI assistant that answers questions based on the provided documents.")
            )
            
            # Add assistant response to chat history
            st.session_state.conversation_history.append({"role": "assistant", "content": response})

# Main function
def main():
    # Render sidebar
    render_sidebar()
    
    # Render chat interface
    render_chat_interface()

# Run the app
if __name__ == "__main__":
    main()
