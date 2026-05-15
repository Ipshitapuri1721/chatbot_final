import os
import re
import json
from pypdf import PdfReader

# =========================
# PDF FOLDER
# =========================

PDF_FOLDER = "data/pdfs"

# =========================
# ALL Q/A STORAGE
# =========================

all_qa = []

# =========================
# READ ALL PDFs
# =========================

for file in os.listdir(PDF_FOLDER):

    if file.endswith(".pdf"):

        path = os.path.join(PDF_FOLDER, file)

        print(f"📘 Reading: {file}")

        try:
            reader = PdfReader(path)

            text = ""

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += "\n" + page_text

            # =========================
            # EXTRACT Q/A
            # =========================

            pattern = r"Q\d*[:.]?\s*(.*?)\nA\d*[:.]?\s*(.*?)(?=\nQ\d*[:.]|\Z)"

            matches = re.findall(pattern, text, re.DOTALL)

            print(f"✅ Found {len(matches)} Q/A")

            for q, a in matches:

                question = q.strip()
                answer = a.strip()

                if len(question) > 5 and len(answer) > 5:

                    all_qa.append({
                        "question": question,
                        "answer": answer
                    })

        except Exception as e:

            print(f"❌ Error in {file}: {e}")

# =========================
# REMOVE DUPLICATES
# =========================

unique_data = []
seen = set()

for item in all_qa:

    key = (
        item["question"].lower(),
        item["answer"].lower()
    )

    if key not in seen:

        seen.add(key)

        unique_data.append(item)

# =========================
# SAVE MASTER DATASET
# =========================

with open("final_master_dataset.json", "w", encoding="utf-8") as f:

    json.dump(unique_data, f, indent=4)

print("\n🎉 FINAL UNIQUE DATASET CREATED")
print("✅ Total Unique Q/A:", len(unique_data))