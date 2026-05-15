from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# EMBEDDING MODEL
# =========================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)


# =========================================================
# LOAD CHROMA DB
# =========================================================

db = Chroma(
    persist_directory="chroma",
    embedding_function=embedding_model
)


# =========================================================
# ASK QUESTION
# =========================================================

query = input("Ask Question: ")


# =========================================================
# SEARCH
# =========================================================

docs = db.similarity_search(
    query,
    k=5
)


# =========================================================
# RESULTS
# =========================================================

print("\n🔎 RESULTS:\n")

for d in docs:

    print("Q:", d.page_content)

    print("A:", d.metadata["answer"])

    print("-" * 60)