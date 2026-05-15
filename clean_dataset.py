import json
import re

INPUT_FILE = "final_clean_dataset.json"
OUTPUT_FILE = "final_fixed_dataset.json"

# =========================================
# LOAD DATA
# =========================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

cleaned_data = []

# =========================================
# CLEAN TEXT
# =========================================

def clean_text(text):

    if not text:
        return ""

    text = str(text)

    # Remove Question: / Answer:
    text = re.sub(r"^Question:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Answer:\s*", "", text, flags=re.IGNORECASE)

    # Remove IDs like A1791:
    text = re.sub(r"A\d+:\s*", "", text)

    # Remove ending numbers like 445.
    text = re.sub(r"\s+\d+\.$", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# =========================================
# SPLIT MERGED QUESTIONS
# =========================================

def split_questions(question_text):

    # Split when another question starts
    parts = re.split(
        r'(?=(?:What|Who|How|When|Where|Why|Can|Is|Are|Does|Do|Tell|Give|Please|May|Could|Would)\b)',
        question_text
    )

    cleaned_parts = []

    for part in parts:

        part = part.strip()

        if len(part) < 5:
            continue

        cleaned_parts.append(part)

    return cleaned_parts

# =========================================
# PROCESS DATA
# =========================================

for item in data:

    question = clean_text(item.get("question", ""))
    answer = clean_text(item.get("answer", ""))

    # Skip empty
    if not question or not answer:
        continue

    # Skip huge corrupted entries
    if len(question) > 300:
        continue

    # Skip broken OCR placement garbage
    bad_patterns = [
        "a179",
        "a178",
        "placement percentage",
        "out of which",
        "academic year"
    ]

    bad = False

    for pattern in bad_patterns:
        if pattern.lower() in question.lower():
            bad = True

    if bad:
        continue

    # =====================================
    # SPLIT MERGED QUESTIONS
    # =====================================

    split_qs = split_questions(question)

    for q in split_qs:

        cleaned_data.append({
            "question": q,
            "answer": answer
        })

# =========================================
# REMOVE DUPLICATES
# =========================================

unique_data = []
seen = set()

for item in cleaned_data:

    key = (
        item["question"].lower(),
        item["answer"].lower()
    )

    if key not in seen:
        seen.add(key)
        unique_data.append(item)

# =========================================
# SAVE
# =========================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(unique_data, f, indent=4, ensure_ascii=False)

print("✅ Dataset cleaned and split successfully")
print(f"📁 Saved as: {OUTPUT_FILE}")
print(f"📊 Total clean Q/A: {len(unique_data)}")