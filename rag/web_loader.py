import requests
from bs4 import BeautifulSoup


def extract_text_from_url(url: str) -> str:
    """
    Extract clean, structured visible text from a webpage.
    Optimized for RAG / chatbot usage.
    """

    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # ❌ Remove unwanted elements
        for tag in soup(["script", "style", "noscript", "header", "footer"]):
            tag.decompose()

        # ✅ Extract text with structure
        text = soup.get_text(separator="\n")

        # ✅ Clean text (important for embeddings)
        lines = []
        for line in text.split("\n"):
            line = line.strip()

            # Remove empty + very small noisy lines
            if len(line) > 30:
                lines.append(line)

        clean_text = "\n".join(lines)

        return clean_text

    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return ""
    clean_text = clean_text.replace("\xa0", " ")