from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="chroma",
    embedding_function=embedding_model
)

docs = db.similarity_search(
    "What is the name of the college?",
    k=3
)

print("\nRESULTS:\n")

for d in docs:
    print(d.page_content)
    print("------")