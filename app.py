import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


def build_embeddings(provider: str, api_key: str):
    """Return embedders for the chosen provider."""
    if provider == "OpenAI":
        os.environ["OPENAI_API_KEY"] = api_key
        return OpenAIEmbeddings()

    # Gemini (Google Generative AI)
    os.environ["GOOGLE_API_KEY"] = api_key
    # Gemini currently exposes only one embedding model
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def build_chat_model(provider: str, api_key: str, model_name: str):
    """Return chat model for the chosen provider."""
    if provider == "OpenAI":
        os.environ["OPENAI_API_KEY"] = api_key
        return ChatOpenAI(model_name=model_name, temperature=0)

    os.environ["GOOGLE_API_KEY"] = api_key
    return ChatGoogleGenerativeAI(model=model_name, temperature=0)


def process_pdf(uploaded_file, embeddings):
    """Create a FAISS vector store from the uploaded PDF."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)

    return FAISS.from_documents(split_docs, embeddings)


def init_chat_chain(provider: str, api_key: str, model_name: str, vectorstore):
    llm = build_chat_model(provider, api_key, model_name)
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


# ---------------------------- Streamlit UI -----------------------------
st.set_page_config(page_title="PDF Chat", layout="wide")

st.title("📄 PDF Chatbot")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")

    provider = st.selectbox("Model Provider", ["OpenAI", "Gemini"], key="provider")

    if provider == "OpenAI":
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            key="openai_api_key",
        )
        openai_models = [
            ("GPT-4.1", "gpt-4.1"),
            ("GPT-4.1 Mini", "gpt-4.1-mini"),
            ("GPT-4.1 Nano", "gpt-4.1-nano"),
            ("GPT-4o", "gpt-4o"),
            ("GPT-4o Mini", "gpt-4o-mini"),
            ("GPT-3.5 Turbo", "gpt-3.5-turbo"),
            ("GPT-3.5 Turbo 16k", "gpt-3.5-turbo-16k"),
            ("OpenAI o1", "o1"),
            ("OpenAI o3", "o3"),
            ("OpenAI o3 Mini", "o3-mini"),
            ("OpenAI o3 Mini High", "o3-mini-high"),
            ("OpenAI o4 Mini", "o4-mini"),
            ("OpenAI o4 Mini High", "o4-mini-high"),
        ]
        model_name = st.selectbox(
            "OpenAI Chat Model",
            options=[(label, value) for label, value in openai_models],
            format_func=lambda x: x[0],
            key="openai_model",
            help="Select from latest OpenAI models. See OpenAI docs for details."
        )[1]
    else:
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIza...",
            key="gemini_api_key",
        )
        gemini_models = [
            ("Gemini 2.5 Flash Preview 04-17", "gemini-2.5-flash-preview-04-17"),
            ("Gemini 2.5 Pro Preview", "gemini-2.5-pro-preview-03-25"),
            ("Gemini 2.0 Flash", "gemini-2.0-flash"),
            ("Gemini 2.0 Flash-Lite", "gemini-2.0-flash-lite"),
            ("Gemini 2.0 Flash Live", "gemini-2.0-flash-live-001"),
            ("Gemini 1.5 Flash", "gemini-1.5-flash"),
            ("Gemini 1.5 Flash-8B", "gemini-1.5-flash-8b"),
            ("Gemini 1.5 Pro", "gemini-1.5-pro"),
            ("Gemini Embedding", "gemini-embedding-exp"),
            ("Imagen 3 (image generation)", "imagen-3.0-generate-002"),
            ("Veo 2 (video generation)", "veo-2.0-generate-001"),
        ]
        model_name = st.selectbox(
            "Gemini Chat Model",
            options=[(label, value) for label, value in gemini_models],
            format_func=lambda x: x[0],
            key="gemini_model",
            help="Select from latest Gemini models. See Google GenAI docs for details."
        )[1]

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", key="pdf_uploader")

    if uploaded_file and api_key:
        with st.spinner("Indexing PDF ..."):
            embeddings = build_embeddings(provider, api_key)
            vectorstore = process_pdf(uploaded_file, embeddings)
            st.session_state["qa_chain"] = init_chat_chain(
                provider, api_key, model_name, vectorstore
            )
            st.success("PDF indexed! Start chatting below 👇")

# Chat interface
qa_chain = st.session_state.get("qa_chain")
chat_history = st.session_state.setdefault("chat_history", [])

if qa_chain:
    user_query = st.chat_input("Ask a question about your PDF")

    if user_query:
        chat_history.append(("user", user_query))
        with st.spinner("Thinking ..."):
            try:
                answer = qa_chain.invoke(user_query)["result"]
            except Exception as e:
                answer = f"Error: {str(e)}\nCheck your API key and model name."
        chat_history.append(("assistant", answer))

    # Render chat
    for role, content in chat_history:
        with st.chat_message(role):
            st.write(content)
else:
    st.info("➡️ Upload a PDF and provide your API key to begin.")