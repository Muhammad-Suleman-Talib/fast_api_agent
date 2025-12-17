# app.py - FastAPI version of the RAG app
# This converts the Streamlit app to a FastAPI backend.
# Run with: uvicorn app:app --reload
# Assumes you have FastAPI and Uvicorn installed: pip install fastapi uvicorn
# Also assumes the same dependencies as the original code (requests, numpy, faiss-cpu, pickle, dotenv)
# I've fixed typos in the code (e.g., 'uttils' -> 'utils', 'embedings_model' -> 'embeddings_model', 'chukking' -> 'chunking', 'retrival' -> 'retrieval', etc.)
# Organized into functions, but kept in one file for simplicity. In a real project, split into modules.
# Added a /query endpoint: POST with JSON {"query": "your question"} to get the response.
# Returns JSON: {"answer": "the generated answer", "retrieved_chunks": ["chunk1", "chunk2", ...]}

import os
import pickle
import numpy as np
import faiss
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()
API_KEY = os.getenv("EURI_AI_API_KEY")

app = FastAPI(title="RAG API", description="Retrieval Augmented Generation API", version="1.0")

class QueryRequest(BaseModel):
    query: str

def chunk_words(text, max_words=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks

def generate_completion(prompt, model="gpt-4.1-nano"):
    url = "https://api.euron.one/api/v1/euri/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": model,
        "max_tokens": 50,
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raise error on bad response
    return response.json()['choices'][0]['message']['content']

def embeddings_model(text, model='text-embedding-3-small'):
    url = "https://api.euron.one/api/v1/euri/embeddings"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": text,
        "model": model
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return np.array(data["data"][0]["embedding"])

def build_prompt(context_chunks, query):
    context = "\n\n".join(context_chunks)
    return f"""Use the following context to answer the question.
    Context: {context}
    Question: {query}
    Answer: """

def load_faiss_index():
    index_path = "faiss_store/index.faiss"
    mapping_path = "faiss_store/chunk_mapping.pkl"

    try:
        if os.path.exists(index_path) and os.path.exists(mapping_path):
            index = faiss.read_index(index_path)
            with open(mapping_path, "rb") as f:
                chunk_mapping = pickle.load(f)

            if index.ntotal != len(chunk_mapping):
                raise ValueError("Index and mapping size mismatch")

            print(f"FAISS index loaded successfully: {index.ntotal} entries")
            return index, chunk_mapping

    except Exception as e:
        print(f"FAISS load failed ({e}), rebuilding index...")

    doc_path = "docs"
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Document file not found: {doc_path}")

    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_words(text)
    chunk_mapping = []

    if not chunks:
        raise ValueError("No chunks generated from the document")

    test_emb = embeddings_model(chunks[0])
    dim = test_emb.shape[0]

    index = faiss.IndexFlatL2(dim)
    embeddings = []

    for chunk in chunks:
        emb = embeddings_model(chunk)
        if emb.shape != (dim,):
            print(f"Skipping chunk with mismatched embedding dimension: expected {dim}, got {emb.shape}")
            continue
        embeddings.append(emb)
        chunk_mapping.append(chunk)

    if not embeddings:
        raise ValueError("No valid embeddings generated")

    embeddings = np.vstack(embeddings)
    index.add(embeddings)

    os.makedirs("faiss_store", exist_ok=True)
    faiss.write_index(index, index_path)

    with open(mapping_path, "wb") as f:
        pickle.dump(chunk_mapping, f)

    print(f"FAISS index rebuilt successfully: {index.ntotal} entries")
    return index, chunk_mapping

def retrieve_chunks(query, index, chunk_mapping, k=3):
    query_vec = embeddings_model(query)
    query_vec = np.array(query_vec).reshape(1, -1)
    
    if query_vec.shape[1] != index.d:
        raise ValueError(f"Query embedding dimension mismatch: expected {index.d}, got {query_vec.shape[1]}")
    
    D, I = index.search(query_vec, k)
    valid_indices = [i for i in I[0] if 0 <= i < len(chunk_mapping)]
    return [chunk_mapping[i] for i in valid_indices]

@app.get("/")
def root():
    return {"message": "Welcome to RAG API. Use POST /query with {'query': 'your question'}"}

@app.post("/query")
def query_rag(request: QueryRequest):
    try:
        index, chunk_mapping = load_faiss_index()
        top_chunks = retrieve_chunks(request.query, index, chunk_mapping)
        prompt = build_prompt(top_chunks, request.query)
        response = generate_completion(prompt)
        return {
            "answer": response,
            "retrieved_chunks": top_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))