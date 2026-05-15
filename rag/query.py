import pickle
import faiss
import numpy as np
import requests
import re

from sentence_transformers import SentenceTransformer

# ==================================================
# LOAD EMBEDDING MODEL
# ==================================================

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

# ==================================================
# LOAD FAISS INDEX
# ==================================================

index = faiss.read_index(
    "index.faiss"
)

# ==================================================
# LOAD DOCUMENTS
# ==================================================

with open("chunks.pkl", "rb") as f:
    docs = pickle.load(f)

# ==================================================
# OLLAMA CONFIG
# ==================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "llama3"

# ==================================================
# CLEAN RESPONSE FUNCTION
# ==================================================

def clean_response(text):

    text = text.strip()

    # Remove unwanted prefixes
    bad_prefixes = [
        "answer:",
        "direct answer:",
        "final answer:",
        "according to the context",
        "based on the context",
        "here is the answer",
        "sure,"
    ]

    lines = text.split("\n")

    cleaned = []

    for line in lines:

        line = line.strip()

        lower = line.lower()

        skip = False

        for prefix in bad_prefixes:

            if lower.startswith(prefix):
                skip = True
                break

        if not skip and line != "":
            cleaned.append(line)

    text = " ".join(cleaned)

    # Remove repeated spaces
    text = re.sub(r"\s+", " ", text)

    # Remove Question/Answer tags if generated
    text = re.sub(r"Question\s*:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Answer\s*:", "", text, flags=re.IGNORECASE)

    return text.strip()

# ==================================================
# ASK QUESTION FUNCTION
# ==================================================

def ask_question(question):

    # ----------------------------------------------
    # CLEAN USER QUESTION
    # ----------------------------------------------

    q = question.lower().strip()

    q = q.replace(
        "ghec",
        "government hydro engineering college"
    )

    # ----------------------------------------------
    # CREATE QUERY EMBEDDING
    # ----------------------------------------------

    query_embedding = model.encode([q])

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    # ----------------------------------------------
    # SEARCH TOP RESULTS
    # ----------------------------------------------

    distances, indices = index.search(
        query_embedding,
        5
    )

    retrieved_context = []

    for idx in indices[0]:

        if idx < len(docs):

            item = docs[idx]

            question_text = item.get(
                "question",
                ""
            ).strip()

            answer_text = item.get(
                "answer",
                ""
            ).strip()

            if question_text and answer_text:

                combined = f"""
Question: {question_text}

Answer: {answer_text}
""".strip()

                retrieved_context.append(combined)

    # ----------------------------------------------
    # FINAL CONTEXT
    # ----------------------------------------------

    context = "\n\n".join(
        retrieved_context
    )

    # ----------------------------------------------
    # DEBUG PRINT
    # ----------------------------------------------

    print("\n===================================")

    print("USER QUESTION:\n")
    print(question)

    print("\nRETRIEVED CONTEXT:\n")
    print(context)

    print("\n===================================\n")

    # ----------------------------------------------
    # STRICT PROMPT
    # ----------------------------------------------

    prompt = f"""
You are a college enquiry chatbot.

Answer ONLY using the provided context.

STRICT RULES:
- Give only the direct answer.
- Keep answer short and professional.
- Do NOT repeat the question.
- Do NOT explain extra things.
- Do NOT say:
  "According to context"
  "Based on information"
  "Here is the answer"
  "Sure"
- If answer is not available, reply EXACTLY:
Sorry, I could not find relevant information.

Context:
{context}

User Question:
{question}

Answer:
"""

    # ----------------------------------------------
    # SEND TO OLLAMA
    # ----------------------------------------------

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "top_p": 0.1,
                    "num_predict": 80
                }
            }
        )

        result = response.json()["response"]

    except Exception as e:

        return f"Error: {str(e)}"

    # ----------------------------------------------
    # CLEAN FINAL RESPONSE
    # ----------------------------------------------

    final_answer = clean_response(result)

    # ----------------------------------------------
    # EMPTY CHECK
    # ----------------------------------------------

    if len(final_answer) < 3:

        return "Sorry, I could not find relevant information."

    # ----------------------------------------------
    # RETURN FINAL ANSWER
    # ----------------------------------------------

    return final_answer