import json
import time
import re
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from rag.query import ask_question

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print("\nLoading embedding model...\n")

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

# =========================================================
# LOAD TEST DATASET
# =========================================================

with open(
    "test_dataset.json",
    "r",
    encoding="utf-8"
) as f:

    test_data = json.load(f)

# =========================================================
# NORMALIZATION FUNCTION
# =========================================================

def normalize(text):

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

# =========================================================
# VARIABLES
# =========================================================

results = []

correct_exact = 0
correct_semantic = 0

total_similarity = 0
total_response_time = 0

SEMANTIC_THRESHOLD = 0.80

# For confusion matrix
y_true = []
y_pred = []

# =========================================================
# START EVALUATION
# =========================================================

print("\n====================================")
print("STARTING RAG MODEL EVALUATION")
print("====================================")

for index, item in enumerate(test_data):

    question = item["question"]

    expected = item["expected_answer"]

    print(f"\nQuestion {index + 1}")

    print(f"Q: {question}")

    # -----------------------------------------------------
    # RESPONSE TIME
    # -----------------------------------------------------

    start_time = time.time()

    generated = ask_question(question)

    end_time = time.time()

    response_time = round(
        end_time - start_time,
        2
    )

    # -----------------------------------------------------
    # NORMALIZE TEXT
    # -----------------------------------------------------

    expected_clean = normalize(expected)

    generated_clean = normalize(generated)

    # -----------------------------------------------------
    # EXACT MATCH
    # -----------------------------------------------------

    exact_match = (
        expected_clean == generated_clean
    )

    if exact_match:
        correct_exact += 1

    # -----------------------------------------------------
    # SEMANTIC SIMILARITY
    # -----------------------------------------------------

    expected_embedding = model.encode([expected])

    generated_embedding = model.encode([generated])

    similarity = cosine_similarity(
        expected_embedding,
        generated_embedding
    )[0][0]

    similarity = float(similarity)

    # -----------------------------------------------------
    # SEMANTIC MATCH
    # -----------------------------------------------------

    semantic_match = (
        similarity >= SEMANTIC_THRESHOLD
    )

    if semantic_match:
        correct_semantic += 1

    # -----------------------------------------------------
    # STORE TRUE/PREDICTED
    # -----------------------------------------------------

    y_true.append(1)

    if semantic_match:
        y_pred.append(1)
    else:
        y_pred.append(0)

    # -----------------------------------------------------
    # UPDATE TOTALS
    # -----------------------------------------------------

    total_similarity += similarity

    total_response_time += response_time

    # -----------------------------------------------------
    # PRINT OUTPUT
    # -----------------------------------------------------

    print(f"Expected Answer : {expected}")

    print(f"Generated Answer: {generated}")

    print(f"Exact Match     : {exact_match}")

    print(f"Semantic Match  : {semantic_match}")

    print(f"Similarity      : {similarity:.4f}")

    print(f"Response Time   : {response_time} sec")

    # -----------------------------------------------------
    # STORE RESULTS
    # -----------------------------------------------------

    results.append({

        "Question": question,

        "Expected Answer": expected,

        "Generated Answer": generated,

        "Exact Match": exact_match,

        "Semantic Match": semantic_match,

        "Similarity Score": round(similarity, 4),

        "Response Time (sec)": response_time

    })

# =========================================================
# CREATE DATAFRAME
# =========================================================

df = pd.DataFrame(results)

# =========================================================
# FINAL METRICS
# =========================================================

total_questions = len(test_data)

exact_accuracy = (
    correct_exact / total_questions
) * 100

semantic_accuracy = (
    correct_semantic / total_questions
) * 100

average_similarity = (
    total_similarity / total_questions
)

average_response_time = (
    total_response_time / total_questions
)

# =========================================================
# PRECISION / RECALL / F1
# =========================================================

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

# =========================================================
# PRINT FINAL RESULTS
# =========================================================

print("\n====================================")
print("FINAL EVALUATION RESULTS")
print("====================================\n")

print(f"Total Questions          : {total_questions}")

print(f"Exact Match Accuracy     : {exact_accuracy:.2f}%")

print(f"Semantic Accuracy        : {semantic_accuracy:.2f}%")

print(f"Average Similarity       : {average_similarity:.4f}")

print(f"Average Response Time    : {average_response_time:.2f} sec")

print(f"Precision                : {precision:.4f}")

print(f"Recall                   : {recall:.4f}")

print(f"F1 Score                 : {f1:.4f}")

# =========================================================
# SAVE RESULTS
# =========================================================

df.to_csv(
    "evaluation_results.csv",
    index=False
)

print("\nCSV Results Saved.")

# =========================================================
# VISUALIZATION 1
# ACCURACY BAR GRAPH
# =========================================================

plt.figure(figsize=(8, 5))

metrics = [
    "Exact",
    "Semantic"
]

values = [
    exact_accuracy,
    semantic_accuracy
]

plt.bar(metrics, values)

plt.ylabel("Accuracy (%)")

plt.title("Accuracy Comparison")

plt.ylim(0, 100)

plt.savefig(
    "accuracy_comparison.png"
)

plt.close()

# =========================================================
# VISUALIZATION 2
# PRECISION / RECALL / F1
# =========================================================

plt.figure(figsize=(8, 5))

metric_names = [
    "Precision",
    "Recall",
    "F1 Score"
]

metric_values = [
    precision,
    recall,
    f1
]

plt.bar(
    metric_names,
    metric_values
)

plt.ylim(0, 1)

plt.ylabel("Score")

plt.title("Precision Recall F1 Score")

plt.savefig(
    "precision_recall_f1.png"
)

plt.close()

# =========================================================
# VISUALIZATION 3
# SEMANTIC SIMILARITY GRAPH
# =========================================================

plt.figure(figsize=(12, 5))

plt.plot(
    df["Similarity Score"],
    marker="o"
)

plt.xlabel("Question Number")

plt.ylabel("Similarity Score")

plt.title("Semantic Similarity Analysis")

plt.grid(True)

plt.savefig(
    "semantic_similarity_graph.png"
)

plt.close()

# =========================================================
# VISUALIZATION 4
# RESPONSE TIME GRAPH
# =========================================================

plt.figure(figsize=(12, 5))

plt.bar(
    range(total_questions),
    df["Response Time (sec)"]
)

plt.xlabel("Question Number")

plt.ylabel("Response Time (sec)")

plt.title("Response Time Analysis")

plt.savefig(
    "response_time_graph.png"
)

plt.close()

# =========================================================
# VISUALIZATION 5
# PIE CHART
# =========================================================

plt.figure(figsize=(6, 6))

correct = correct_semantic

incorrect = total_questions - correct

plt.pie(
    [correct, incorrect],
    labels=["Correct", "Incorrect"],
    autopct="%1.1f%%"
)

plt.title("Semantic Accuracy Distribution")

plt.savefig(
    "semantic_accuracy_pie_chart.png"
)

plt.close()

# =========================================================
# VISUALIZATION 6
# SIMILARITY DISTRIBUTION
# =========================================================

plt.figure(figsize=(10, 5))

plt.hist(
    df["Similarity Score"],
    bins=10
)

plt.xlabel("Similarity Score")

plt.ylabel("Frequency")

plt.title("Similarity Score Distribution")

plt.savefig(
    "similarity_distribution.png"
)

plt.close()

# =========================================================
# VISUALIZATION 7
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(6, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Incorrect", "Correct"]
)

disp.plot()

plt.title("Confusion Matrix")

plt.savefig(
    "confusion_matrix.png"
)

plt.close()

# =========================================================
# VISUALIZATION 8
# EXACT MATCH DISTRIBUTION
# =========================================================

plt.figure(figsize=(6, 6))

exact_correct = correct_exact

exact_incorrect = total_questions - exact_correct

plt.pie(
    [exact_correct, exact_incorrect],
    labels=["Exact Match", "Mismatch"],
    autopct="%1.1f%%"
)

plt.title("Exact Match Distribution")

plt.savefig(
    "exact_match_distribution.png"
)

plt.close()

# =========================================================
# DONE
# =========================================================

print("\n====================================")
print("EVALUATION COMPLETED SUCCESSFULLY")
print("====================================")

print("\nGenerated Files:")

print("1. evaluation_results.csv")

print("2. accuracy_comparison.png")

print("3. precision_recall_f1.png")

print("4. semantic_similarity_graph.png")

print("5. response_time_graph.png")

print("6. semantic_accuracy_pie_chart.png")

print("7. similarity_distribution.png")

print("8. confusion_matrix.png")

print("9. exact_match_distribution.png")