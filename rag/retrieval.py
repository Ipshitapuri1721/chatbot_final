import faiss
import pickle
import numpy as np

from rag.embedding import model

# =========================
# LOAD FAISS
# =========================

index = faiss.read_index("faiss_index/index.faiss")

with open("faiss_index/chunks.pkl", "rb") as f:
    docs = pickle.load(f)

# =========================
# RETRIEVAL
# =========================

def retrieve(query, top_k=1):

    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for idx in indices[0]:

        if idx < len(docs):
            results.append(docs[idx])

    return results