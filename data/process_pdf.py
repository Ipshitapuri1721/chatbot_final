import re
import json

with open("data/clean_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Normalize formats
text = text.replace("Q:", "Question:")
text = text.replace("A:", "Answer:")

# Remove separators
text = re.sub(r'-{3,}', '\n', text)

# Extract Q&A pairs
pattern = r'Question[:\s]*(.*?)\s*Answer[:\s]*(.*?)(?=Question|\Z)'

matches = re.findall(pattern, text, re.DOTALL)

dataset = []

for q, a in matches:
    q = q.strip()
    a = a.strip()

    # Skip empty or very short
    if len(q) < 5 or len(a) < 5:
        continue

    dataset.append({
        "instruction": q,
        "input": "",
        "output": a
    })

# Save JSON
with open("data/final_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print(f"✅ Extracted {len(dataset)} Q&A pairs")