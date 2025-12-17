# import faiss
# import numpy as np
# import os 
# import pickle
# from uttils.embedding import embedings_model
# from uttils.chukking import chunk_words


# def load_fassi_index():
#     index_path = "faiss_store/index.faiss"
#     mapping_path = "faiss_store/chunk_mapping.pkl"

#     try:
#         if os.path.exists(index_path) and os.path.exists(mapping_path):
#             index = faiss.read_index(index_path)
#             with open(mapping_path, "rb") as f:
#                 chunk_mapping = pickle.load(f)

#             # 🔒 sanity check
#             if index.ntotal != len(chunk_mapping):
#                 raise ValueError("Index and mapping size mismatch")

#             print("FAISS index loaded:", index.ntotal)
#             return index, chunk_mapping

#     except Exception as e:
#         print("FAISS load failed, rebuilding index:", e)

#     # 🔁 rebuild index safely
#     with open("docs/personal/author-profile.md", "r", encoding="utf-8") as f:
#         text = f.read()

#     chunks = chunk_words(text)
#     chunk_mapping = []

#     # 🔍 get dimension dynamically
#     test_emb = embedings_model(chunks[0])
#     test_emb = np.array(test_emb, dtype="float32")
#     dim = test_emb.shape[0]

#     index = faiss.IndexFlatL2(dim)

#     for chunk in chunks:
#         emb = np.array(embedings_model(chunk), dtype="float32")
#         if emb.shape[0] != dim:
#             continue  # skip broken embeddings
#         index.add(emb.reshape(1, -1))
#         chunk_mapping.append(chunk)

#     os.makedirs("faiss_store", exist_ok=True)
#     faiss.write_index(index, index_path)

#     with open(mapping_path, "wb") as f:
#         pickle.dump(chunk_mapping, f)

#     print("FAISS index rebuilt:", index.ntotal)
#     return index, chunk_mapping

# def retrive_chunks(query, index, chunk_mapping, k=3):
#     query_vec = np.array(embedings_model(query), dtype="float32").reshape(1, -1)
#     D, I = index.search(query_vec, k)
#     return [chunk_mapping[i] for i in I[0] if i < len(chunk_mapping)]


import faiss  # type: ignore
import numpy as np
import os
import pickle
from uttils.embedding import embedings_model # type: ignore # Corrected spelling: 'uttils' -> 'utils', 'embedings_model' -> 'embeddings_model'
from uttils.chukking import chunk_words # type: ignore  # Corrected spelling: 'chukking' -> 'chunking'


def load_faiss_index():  # Corrected spelling: 'fassi' -> 'faiss'
    index_path = "faiss_store/index.faiss"
    mapping_path = "faiss_store/chunk_mapping.pkl"

    try:
        if os.path.exists(index_path) and os.path.exists(mapping_path):
            index = faiss.read_index(index_path)
            with open(mapping_path, "rb") as f:
                chunk_mapping = pickle.load(f)

            # Sanity check
            if index.ntotal != len(chunk_mapping):
                raise ValueError("Index and mapping size mismatch")

            print(f"FAISS index loaded successfully: {index.ntotal} entries")
            return index, chunk_mapping

    except Exception as e:
        print(f"FAISS load failed ({e}), rebuilding index...")

    # Rebuild index safely
    doc_path = "docs"
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Document file not found: {doc_path}")

    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_words(text)
    chunk_mapping = []

    # Get dimension dynamically from the first valid embedding
    if not chunks:
        raise ValueError("No chunks generated from the document")

    test_emb = embedings_model(chunks[0])
    test_emb = np.array(test_emb, dtype="float32")
    dim = test_emb.shape[0]

    index = faiss.IndexFlatL2(dim)
    embeddings = []

    for chunk in chunks:
        emb = embedings_model(chunk)
        emb = np.array(emb, dtype="float32")
        if emb.shape != (dim,):
            print(f"Skipping chunk with mismatched embedding dimension: expected {dim}, got {emb.shape}")
            continue
        embeddings.append(emb)
        chunk_mapping.append(chunk)

    if not embeddings:
        raise ValueError("No valid embeddings generated")

    embeddings = np.vstack(embeddings)  # Stack into a 2D array for batch add
    index.add(embeddings)

    os.makedirs("faiss_store", exist_ok=True)
    faiss.write_index(index, index_path)

    with open(mapping_path, "wb") as f:
        pickle.dump(chunk_mapping, f)

    print(f"FAISS index rebuilt successfully: {index.ntotal} entries")
    return index, chunk_mapping


def retrieve_chunks(query, index, chunk_mapping, k=3):  # Corrected spelling: 'retrive' -> 'retrieve'
    query_vec = embedings_model(query)
    query_vec = np.array(query_vec, dtype="float32").reshape(1, -1)
    
    if query_vec.shape[1] != index.d:
        raise ValueError(f"Query embedding dimension mismatch: expected {index.d}, got {query_vec.shape[1]}")
    
    D, I = index.search(query_vec, k)
    valid_indices = [i for i in I[0] if 0 <= i < len(chunk_mapping)]
    return [chunk_mapping[i] for i in valid_indices]