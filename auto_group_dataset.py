import json
import os

INPUT_FILE = "final_master_dataset.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print("✅ Total Records:", len(dataset))

categories = {
    "clubs": [
        "club",
        "society",
        "nss",
        "sports",
        "literary",
        "cultural",
        "yoga",
        "fine arts"
    ],

    "hostel": [
        "hostel",
        "mess",
        "room",
        "warden",
        "wifi"
    ],

    "faculty": [
        "faculty",
        "teacher",
        "professor",
        "hod",
        "director",
        "principal",
        "coordinator"
    ],

    "courses": [
        "course",
        "btech",
        "mtech",
        "branch",
        "department",
        "semester"
    ],

    "admission": [
        "admission",
        "jee",
        "counselling",
        "eligibility",
        "lateral entry"
    ],

    "placements": [
        "placement",
        "package",
        "internship",
        "career"
    ],

    "facilities": [
        "library",
        "lab",
        "canteen",
        "transport",
        "bus",
        "facility",
        "wifi"
    ]
}

os.makedirs("data/grouped", exist_ok=True)

grouped_data = {}

for category in categories:
    grouped_data[category] = []

for item in dataset:

    question = item.get("question", "").strip()
    answer = item.get("answer", "").strip()

    combined_text = f"""
Question: {question}

Answer: {answer}
"""

    assigned = False

    for category, keywords in categories.items():

        for keyword in keywords:

            if keyword.lower() in question.lower():

                grouped_data[category].append({
                    "text": combined_text
                })

                assigned = True
                break

        if assigned:
            break

for category, content in grouped_data.items():

    output = {
        "category": category,
        "content": content
    }

    path = f"data/grouped/{category}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"✅ Saved: {path}")

print("\n🎉 ALL CATEGORY FILES CREATED")