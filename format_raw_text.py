import re

input_file = "data/processed/raw_text.txt"
output_file = "data/processed/clean_text.txt"

with open(input_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

clean_data = []

current_q = None

for line in lines:
    
    # Match Question like Q1:, Q23:
    if re.match(r"Q\d+:", line):
        current_q = re.sub(r"Q\d+:\s*", "", line).strip()

    # Match Answer like A1:, A23:
    elif re.match(r"A\d+:", line):
        answer = re.sub(r"A\d+:\s*", "", line).strip()

        if current_q:
            clean_data.append(f"Question: {current_q}\nAnswer: {answer}\n")
            current_q = None

# Save cleaned data
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(clean_data))


print(f"Total lines read:{len(lines)}")
print(f"✅ Extracted {len(clean_data)} QA pairs")