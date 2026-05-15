from rag.llm import generate_answer

query = "Where is GHEC located?"
context_chunks = [
    "Answer: GHEC (Government Hydro Engineering College Bandla) is located in Bandla, Bilaspur district, Himachal Pradesh, India."
]

print("Question:")
print(query)
print("\nAnswer:")
print(generate_answer(query, context_chunks))