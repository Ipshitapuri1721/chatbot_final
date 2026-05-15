import os
import json
import re

GROUPED_FOLDER = "data/grouped"

for filename in os.listdir(GROUPED_FOLDER):

    if not filename.endswith(".json"):
        continue

    path = os.path.join(GROUPED_FOLDER, filename)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = []

    # Handle old format
    if isinstance(data, dict):
        records = data.get("content", [])
    else:
        records = data

    for item in records:

        # Get raw text
        if "text" in item:
            text = item["text"]
        else:
            question = item.get("question", "")
            answer = item.get("answer", "")
            text = f"Question: {question}\nAnswer: {answer}"

        # Convert escaped newlines
        text = text.replace("\\n", " ")

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text)

        # Extract question and answer
        q_match = re.search(r"Question:(.*?)Answer:", text, re.DOTALL)
        a_match = re.search(r"Answer:(.*)", text, re.DOTALL)

        if q_match and a_match:

            question = q_match.group(1).strip()
            answer = a_match.group(1).strip()

            # Final cleanup
            question = re.sub(r"\s+", " ", question)
            answer = re.sub(r"\s+", " ", answer)

            cleaned.append({
                "question": question,
                "answer": answer
            })

    # Save cleaned file
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=4, ensure_ascii=False)

    print(f"✅ Cleaned: {filename}")

print("\n🎉 ALL FILES FULLY CLEANED")