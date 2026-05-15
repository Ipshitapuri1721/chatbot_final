# ==============================
# FINAL POLISHED RAG PIPELINE
# ==============================

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from sentence_transformers import CrossEncoder


# ==============================
# CONFIGURATION
# ==============================

PERSIST_DIRECTORY = "chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PDF_COLLECTION = "pdf_docs"
WEB_COLLECTION = "web_docs"


# ==============================
# LOAD EMBEDDINGS
# ==============================

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


# ==============================
# LOAD VECTOR DATABASE
# ==============================

pdf_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
    collection_name=PDF_COLLECTION
)

web_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
    collection_name=WEB_COLLECTION
)


# ==============================
# RETRIEVERS
# ==============================

pdf_vector = pdf_store.as_retriever(search_kwargs={"k": 2})
web_vector = web_store.as_retriever(search_kwargs={"k": 2})


# ==============================
# BM25 (KEYWORD SEARCH)
# ==============================

pdf_docs = pdf_store.get().get("documents", [])
web_docs = web_store.get().get("documents", [])

pdf_bm25 = BM25Retriever.from_documents(
    [Document(page_content=d) for d in pdf_docs]
) if pdf_docs else None

web_bm25 = BM25Retriever.from_documents(
    [Document(page_content=d) for d in web_docs]
) if web_docs else None

if pdf_bm25:
    pdf_bm25.k = 2

if web_bm25:
    web_bm25.k = 2


# ==============================
# LLM (UPGRADE THIS!)
# ==============================

llm = OllamaLLM(
    model="llama3",   # 🔥 change from tinyllama
    temperature=0
)


# ==============================
# RERANKER
# ==============================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ==============================
# CLEAN CONTEXT
# ==============================

def clean_context(text):
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()

        # remove Q/A noise
        if line.startswith("Q") or line.startswith("A"):
            continue

        if len(line) > 20:
            cleaned.append(line)

    return "\n".join(cleaned)


# ==============================
# RERANK FUNCTION
# ==============================

def rerank(question, docs):
    pairs = [[question, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked]


# ==============================
# CONFIDENCE CHECK
# ==============================

def low_confidence(context):
    return len(context.strip()) < 50


# ==============================
# MAIN RAG PIPELINE
# ==============================

def rag_pipeline(question: str):

    try:

        # -------------------------
        # RETRIEVAL
        # -------------------------

        pdf_docs_vec = pdf_vector.invoke(question)
        web_docs_vec = web_vector.invoke(question)

        pdf_docs_kw = pdf_bm25.invoke(question) if pdf_bm25 else []
        web_docs_kw = web_bm25.invoke(question) if web_bm25 else []

        all_docs = pdf_docs_vec + web_docs_vec + pdf_docs_kw + web_docs_kw

        # remove duplicates
        unique_docs = list({
            doc.page_content: doc for doc in all_docs
        }.values())

        if not unique_docs:
            return {
                "answer": "Information not available in provided documents.",
                "sources": []
            }

        # -------------------------
        # RERANK
        # -------------------------

        ranked_docs = rerank(question, unique_docs)

        # -------------------------
        # BUILD CONTEXT
        # -------------------------

        top_docs = ranked_docs[:2]

        context = "\n\n".join(doc.page_content for doc in top_docs)
        context = clean_context(context)

        # -------------------------
        # CONFIDENCE CHECK
        # -------------------------

        if low_confidence(context):
            return {
                "answer": "Information not available in provided documents.",
                "sources": []
            }

        # -------------------------
        # PROMPT
        # -------------------------

        prompt = f"""
You are a smart college enquiry assistant.

Answer the question using ONLY the context below.

RULES:
- Do NOT copy text directly.
- Do NOT include Q&A numbers.
- Keep answer short and clear.
- Combine information if needed.
- If answer is missing, say:
  "Information not available in provided documents."

Context:
{context}

Question:
{question}

Final Answer:
"""

        # -------------------------
        # GENERATE ANSWER
        # -------------------------

        response = llm.invoke(prompt).strip()

        # -------------------------
        # SOURCES
        # -------------------------

        sources = list(set(
            doc.metadata.get("source", "unknown")
            for doc in top_docs
        ))

        return {
            "answer": response,
            "sources": sources
        }

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }