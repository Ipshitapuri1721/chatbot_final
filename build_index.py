import fitz
import faiss
import pickle
import numpy as np
import re

from sentence_transformers import SentenceTransformer

# ==================================================
# LOAD EMBEDDING MODEL
# ==================================================

print("\nLoading embedding model...\n")

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

# ==================================================
# PDF FILES
# ==================================================

pdf_files = [
    "data/pdfs/final0.2.pdf",
    "data/pdfs/Dataset-1.pdf",
    "data/pdfs/Dataset_augmented_sp.pdf"
]

# ==================================================
# EXTRACT TEXT FROM PDFS
# ==================================================

full_text = ""

for pdf in pdf_files:

    print(f"Reading PDF: {pdf}")

    doc = fitz.open(pdf)

    for page in doc:

        text = page.get_text("text")

        if text:
            full_text += "\n" + text

    doc.close()

# ==================================================
# CLEAN RAW TEXT
# ==================================================

full_text = full_text.replace("\n", " ")

full_text = re.sub(
    r"\s+",
    " ",
    full_text
)

# ==================================================
# FIX BROKEN WORDS
# ==================================================

full_text = full_text.replace("Engin eering", "Engineering")
full_text = full_text.replace("postgradua te", "postgraduate")
full_text = full_text.replace("w ho", "who")
full_text = full_text.replace("sh ould", "should")

# ==================================================
# EXTRACT Q/A PAIRS
# ==================================================

pattern = r"Q\d+:\s*(.*?)\s*A\d+:\s*(.*?)(?=\s*Q\d+:|$)"

matches = re.findall(
    pattern,
    full_text,
    re.DOTALL
)

documents = []

# ==================================================
# PROCESS EACH Q/A
# ==================================================

for question, answer in matches:

    # ----------------------------------------------
    # CLEAN QUESTION
    # ----------------------------------------------

    question = question.strip()

    question = re.sub(
        r"\s+",
        " ",
        question
    )

    if not question.endswith("?"):
        question += "?"

    # ----------------------------------------------
    # CLEAN ANSWER
    # ----------------------------------------------

    answer = answer.strip()

    answer = re.sub(
        r"\s+",
        " ",
        answer
    )

    # Remove accidental next question fragments
    answer = re.split(
        r"\s*Q\d+:",
        answer
    )[0]

    # Remove accidental answer numbers
    answer = re.sub(
        r"A\d+\s*:",
        "",
        answer
    )

    answer = answer.strip()

    # ----------------------------------------------
    # SKIP BAD DATA
    # ----------------------------------------------

    if len(question) < 5:
        continue

    if len(answer) < 5:
        continue

    # Skip incomplete answers
    incomplete_patterns = [
        "was established in the",
        "offers undergraduate",
        "the college offers undergraduate",
        "duration of b.tech",
        "postgraduate (m.tech"
    ]

    bad = False

    for p in incomplete_patterns:

        if answer.lower().endswith(p):
            bad = True
            break

    if bad:
        continue

    # ----------------------------------------------
    # FINAL TEXT
    # ----------------------------------------------

    combined = f"""
Question: {question}

Answer: {answer}
""".strip()

    documents.append({
        "question": question,
        "answer": answer,
        "text": combined
    })

# ==================================================
# REMOVE DUPLICATES
# ==================================================

unique_documents = []

seen = set()

for doc in documents:

    key = (
        doc["question"].lower(),
        doc["answer"].lower()
    )

    if key not in seen:

        seen.add(key)

        unique_documents.append(doc)

documents = unique_documents

# ==================================================
# SHOW SAMPLE DATA
# ==================================================

print("\n===================================")
print("SAMPLE EXTRACTED DATA")
print("===================================\n")

for i in range(min(10, len(documents))):

    print(documents[i])
    print()

# ==================================================
# CREATE EMBEDDINGS
# ==================================================

texts = [
    doc["text"]
    for doc in documents
]

print("\nCreating embeddings...\n")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

embeddings = np.array(
    embeddings,
    dtype="float32"
)

# ==================================================
# CREATE FAISS INDEX
# ==================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(embeddings)

# ==================================================
# SAVE INDEX + DOCUMENTS
# ==================================================

faiss.write_index(
    index,
    "index.faiss"
)

with open(
    "chunks.pkl",
    "wb"
) as f:

    pickle.dump(
        documents,
        f
    )

# ==================================================
# DONE
# ==================================================

print("\n===================================")
print("FAISS INDEX CREATED SUCCESSFULLY")
print("===================================")

print(f"\nTotal Q/A pairs indexed: {len(documents)}")