# ==============================
# WEBSITE INGESTION SCRIPT (FINAL CLEAN VERSION)
# ==============================

import requests
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ==============================
# CONFIG
# ==============================
PERSIST_DIRECTORY = "chroma"
COLLECTION_NAME = "web_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BASE_URL = "https://ghec.ac.in/"


# ==============================
# LOAD EMBEDDINGS
# ==============================
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


# ==============================
# TEXT SPLITTER
# ==============================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)


# ==============================
# GET ALL INTERNAL LINKS
# ==============================
def get_all_links(base_url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(base_url, headers=headers, timeout=10)

        soup = BeautifulSoup(response.text, "html.parser")

        links = set()

        for a in soup.find_all("a", href=True):
            link = a["href"]

            # Convert relative links to absolute
            if link.startswith("/"):
                link = base_url.rstrip("/") + link

            # Keep only internal links
            if base_url in link:
                links.add(link)

        return list(links)

    except Exception as e:
        print("❌ Error fetching links:", e)
        return []


# ==============================
# SCRAPE WEBSITE FUNCTION
# ==============================
def scrape_website(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted tags
        for tag in soup(["script", "style", "noscript", "header", "footer"]):
            tag.decompose()

        text = soup.get_text(separator="\n")

        # Clean text
        lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 30]

        return "\n".join(lines)

    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return ""


# ==============================
# MAIN INGEST FUNCTION
# ==============================
def ingest_web():

    print("🚀 Starting website ingestion...")

    all_documents = []
    visited = set()

    # Step 1: Get all website links
    urls = get_all_links(BASE_URL)

    print(f"🔗 Total links found: {len(urls)}")

    # Step 2: Loop through each page
    for url in urls:

        if url in visited:
            continue

        visited.add(url)

        print(f"🌐 Scraping: {url}")

        content = scrape_website(url)

        if content:
            chunks = text_splitter.split_text(content)

            for chunk in chunks:
                all_documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": url,
                            "type": "website"
                        }
                    )
                )

    if not all_documents:
        print("❌ No documents scraped.")
        return

    print(f"✅ Total Web Chunks Created: {len(all_documents)}")

    # ------------------------------
    # Delete old collection
    # ------------------------------
    try:
        existing_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        existing_store.delete_collection()
        print("🗑 Old web_docs collection deleted.")
    except:
        pass

    # ------------------------------
    # Store in Chroma
    # ------------------------------
    Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME
    )

    print("🎉 Website ingestion complete!")


# ==============================
# RUN SCRIPT
# ==============================
if __name__ == "__main__":
    ingest_web()