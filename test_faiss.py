import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer


# ==========================================
# LOAD SAME MODEL USED DURING INDEXING
# ==========================================

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

# ==========================================
# LOAD FAISS INDEX
# ==========================================

index = faiss.read_index(
    "faiss_index/index.faiss"
)

# ==========================================
# LOAD DOCUMENTS
# ==========================================

with open(
    "faiss_index/chunks.pkl",
    "rb"
) as f:

    docs = pickle.load(f)

# ==========================================
# ASK QUESTION
# ==========================================

query = input("Ask Question: ")

# ==========================================
# CREATE QUERY EMBEDDING
# ==========================================

query_embedding = model.encode(
    [query]
)

query_embedding = np.array(
    query_embedding,
    dtype="float32"
)

# ==========================================
# SEARCH
# ==========================================

distances, indices = index.search(
    query_embedding,
    5
)

# ==========================================
# RESULTS
# ==========================================

print("\n🔎 RESULTS:\n")

for idx in indices[0]:

    item = docs[idx]

    print("Q:", item["question"])
    print("A:", item["answer"])
    print("-" * 60)