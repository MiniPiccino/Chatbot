# -*- coding: utf-8 -*-

import streamlit as st
import logging
from PIL import Image, ImageEnhance
import time
import json
import requests
from transformers import pipeline
import base64
from dotenv import load_dotenv
import os
import PyPDF2
import chromadb
import asyncio
load_dotenv() 

try:
    from openai import OpenAI, OpenAIError
except ImportError:
    OpenAI = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

# Configure logging
logging.basicConfig(level=logging.INFO)

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
def main():
    st.title("Document Chatbot")

    # Model selection
    MODEL_OPTIONS = [
        "OpenAI GPT-4o (cloud)",
        "DeepSeek R1 (cloud)", 
        "HuggingFace DialoGPT (cloud)",
        "GPT2 (local)",
        "DistilGPT2 (local)",
        "Llama3 (local)"
    ]
    selected_model = st.sidebar.selectbox("Select Model:", MODEL_OPTIONS, index=1)

    # Upload document
    # uploaded_file = st.sidebar.file_uploader("Upload a document", type=["txt", "pdf"])
    # doc_text = ""
    # if uploaded_file is not None:
    #     if uploaded_file.type == "application/pdf":
    #         doc_text = extract_pdf_text(uploaded_file)
    #     elif uploaded_file.type == "text/plain":
    #         doc_text = uploaded_file.read().decode("utf-8")
    #     st.sidebar.success(f"Uploaded: {uploaded_file.name}")
    #     st.session_state["uploaded_doc_text"] = doc_text

    #     # Add to ChromaDB (if not already added)
    #     add_uploaded_doc_to_chromadb(doc_text)
    st.sidebar.success("Using preloaded ChromaDB collection: 'poslovniModeli'")

    # Chat interface
    if "history" not in st.session_state:
        st.session_state.history = []

    chat_input = st.chat_input("Ask a question about your document:")
    #if chat_input and doc_text:
    if chat_input:
        # Get relevant context from ChromaDB
        chromadb_context = get_chromadb_context(chat_input, top_k=3)
        # Build prompt
        prompt = (
            f"Use the following context to answer the user's question:\n\n"
            f"{chromadb_context}\n\n"
            f"User question: {chat_input}\n"
            f"Answer:"
        )
        # Get model response
        response = get_model_response(prompt, selected_model)
        st.session_state.history.append({"role": "user", "content": chat_input})
        st.session_state.history.append({"role": "assistant", "content": response})
        with st.sidebar.expander("Last context from ChromaDB"):
            st.write(chromadb_context)
    # Display chat history
    for message in st.session_state.history[-20:]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Helper: Add uploaded doc to ChromaDB
def add_uploaded_doc_to_chromadb(doc_text):
    import chromadb
    import asyncio
    async def _add():
        chroma_client = await chromadb.AsyncHttpClient()
        collection = await chroma_client.get_or_create_collection(name="poslovniModeli")
        chunk_size = 1000
        overlap = 100
        chunks = []
        start = 0
        while start < len(doc_text):
            end = start + chunk_size
            chunks.append(doc_text[start:end])
            start += chunk_size - overlap
        ids = [f"uploaded_{i}" for i in range(len(chunks))]
        BATCH_SIZE = 5000
        for i in range(0, len(chunks), BATCH_SIZE):
            await collection.add(
                ids=ids[i:i+BATCH_SIZE],
                documents=chunks[i:i+BATCH_SIZE],
            )
    asyncio.run(_add())

def get_top_chunks(results, top_k=3):
    # Get the top_k most relevant chunks (lowest distances)
    docs = results['documents'][0]
    distances = results['distances'][0]
    # Pair and sort by distance
    sorted_chunks = sorted(zip(docs, distances), key=lambda x: x[1])
    # Return the top_k chunks as a single string
    return "\n".join(chunk for chunk, _ in sorted_chunks[:top_k])

# Helper: Get context from ChromaDB
# def get_chromadb_context(query, top_k=3):
#     import chromadb
#     import asyncio
#     async def _query():
#         chroma_client = await chromadb.AsyncHttpClient()
#         collection = await chroma_client.get_or_create_collection(name="poslovniModeli") 
#         results = await collection.query(query_texts=[query])
#         docs = results['documents'][0]
#         distances = results['distances'][0]
#         sorted_chunks = sorted(zip(docs, distances), key=lambda x: x[1])
#         return "\n".join(chunk for chunk, _ in sorted_chunks[:top_k])
#     return asyncio.run(_query())
def get_chromadb_context(query, top_k=3):
    import chromadb
    import asyncio

    async def _query():
        chroma_client = await chromadb.AsyncHttpClient()
        collection = await chroma_client.get_or_create_collection(name="poslovniModeli")
        results = await collection.query(query_texts=[query])
        return get_top_chunks(results, top_k=top_k)

    return asyncio.run(_query())

# Helper: Extract PDF text
def extract_pdf_text(uploaded_file):
    import PyPDF2
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Helper: Get model response
def get_model_response(prompt, model_choice):
    if model_choice == "GPT2 (local)":
        hf_chatbot = get_hf_pipeline("gpt2")
        result = hf_chatbot(prompt)
        return result[0]['generated_text'] if result and 'generated_text' in result[0] else "No response."
    # Add other model logic as needed...
    elif model_choice == "DeepSeek R1 (cloud)":
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        try:
            completion = client.chat.completions.create(
                model="deepseek/deepseek-r1-0528:free",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5000,
                temperature=0.7,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"DeepSeek error: {str(e)}"
    return "Model not implemented."

if __name__ == "__main__":
    main()