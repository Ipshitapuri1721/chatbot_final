from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    return chunks
