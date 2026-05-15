import json

input_file = "clean_dataset.jsonl"   # your jsonl file
output_file = "cleaned_data.json"  # output json file

data = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print("Conversion done ✅")