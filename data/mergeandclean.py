import json
import re
from sentence_transformers import SentenceTransformer, util

# ---------- Helper: Load JSON or JSONL ----------
def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)  # Try normal JSON
        except:
            f.seek(0)
            for line in f:      # Fallback JSONL
                data.append(json.loads(line))
    return data

# ---------- Helper: Normalize questions ----------
def normalize(q):
    q = q.lower()
    q = re.sub(r"(can you tell me|do you know|may i ask|please tell me|i want to know|is it true that)", "", q)
    q = re.sub(r'\s+', ' ', q)
    return q.strip()

# ---------- Load datasets ----------
data1 = load_data("clean_dataset.jsonl")
data2 = load_data("data/final_dataset.json")

# ---------- Merge ----------
all_data = data1 + data2
print(f"Total before cleaning: {len(all_data)}")

# ---------- Model ----------
model = SentenceTransformer("all-MiniLM-L6-v2")

clean_data = []
questions = []

SIMILARITY_THRESHOLD = 0.85

# ---------- Deduplicate ----------
for item in all_data:
    question = normalize(item.get("instruction", ""))

    if not question:
        continue

    if len(questions) == 0:
        questions.append(question)
        clean_data.append(item)
        continue

    emb1 = model.encode([question])
    emb2 = model.encode(questions)

    similarity = util.cos_sim(emb1, emb2)[0]

    if max(similarity) < SIMILARITY_THRESHOLD:
        questions.append(question)
        clean_data.append(item)

# ---------- Save ----------
with open("data/final_clean_dataset.json", "w", encoding="utf-8") as f:
    json.dump(clean_data, f, indent=2)

print(f"✅ Final dataset size: {len(clean_data)}")