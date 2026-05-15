import os
import json
import faiss
import pickle
import numpy as np
import fitz

from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# =========================
# FOLDERS
# =========================

JSON_FOLDER = "data/grouped"
PDF_FOLDER = "data/pdfs"

# =========================
# LOAD DATA
# =========================

all_data = []

print("\nLoading JSON files...\n")

for file in os.listdir(JSON_FOLDER):

    if file.endswith(".json"):

        path = os.path.join(JSON_FOLDER, file)

        with open(path, "r", encoding="utf-8") as f:

            data = json.load(f)

            all_data.extend(data)

            print("Loaded:", file)

# =========================
# READ PDF FILES
# =========================

print("\nReading PDFs...\n")

for file in os.listdir(PDF_FOLDER):

    if file.endswith(".pdf"):

        path = os.path.join(PDF_FOLDER, file)

        print("Reading:", file)

        doc = fitz.open(path)

        text = ""

        for page in doc:

            text += page.get_text()

        doc.close()

        text = text.replace("\n", " ")
        text = " ".join(text.split())

        chunk_size = 500

        for i in range(0, len(text), chunk_size):

            chunk = text[i:i + chunk_size]

            if len(chunk.strip()) > 100:

                all_data.append({
                    "question": chunk,
                    "answer": chunk
                })

print("\nTotal Records:", len(all_data))

# =========================
# QUESTIONS
# =========================

questions = []

for item in all_data:

    questions.append(item["question"])

# =========================
# EMBEDDING MODEL
# =========================

print("\nLoading embedding model...\n")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# CREATE EMBEDDINGS
# =========================

print("\nCreating embeddings...\n")

embeddings = model.encode(
    questions,
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")

# =========================
# FAISS
# =========================

print("\nCreating FAISS index...\n")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

os.makedirs("faiss_index", exist_ok=True)

faiss.write_index(
    index,
    "faiss_index/index.faiss"
)

with open(
    "faiss_index/chunks.pkl",
    "wb"
) as f:

    pickle.dump(all_data, f)

print("FAISS saved")

# =========================
# CHROMADB
# =========================

print("\nCreating ChromaDB...\n")

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = []

for item in all_data:

    doc = Document(
        page_content=item["question"],
        metadata={
            "answer": item["answer"]
        }
    )

    documents.append(doc)

db = Chroma.from_documents(
    documents=documents,
    embedding=embedding_function,
    persist_directory="chroma"
)

print("ChromaDB saved")

print("\nALL DONE SUCCESSFULLY")