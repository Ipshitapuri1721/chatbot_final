from fastapi import FastAPI
from pydantic import BaseModel
from rag.query import ask_question

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(q: Query):
    answer = ask_question(q.question)
    return {"answer": answer}