import json
from sentence_transformers import SentenceTransformer, util

input_file = "data/train.json"
output_file = "clean_dataset.jsonl"

model = SentenceTransformer("all-MiniLM-L6-v2")

questions = []
clean_data = []

SIMILARITY_THRESHOLD = 0.85

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

    for item in data:
        question = item.get("instruction", "")

        if not question:
            continue

        if len(questions) == 0:
            questions.append(question)
            clean_data.append(item)
        else:
            emb1 = model.encode([question])
            emb2 = model.encode(questions)

            sim = util.cos_sim(emb1, emb2)[0]

            if max(sim) < SIMILARITY_THRESHOLD:
                questions.append(question)
                clean_data.append(item)

# Save
with open(output_file, "w", encoding="utf-8") as f:
    for item in clean_data:
        f.write(json.dumps(item) + "\n")

print(f"Final dataset size: {len(clean_data)}")