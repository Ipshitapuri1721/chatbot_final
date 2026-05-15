import os
import time
import requests
import ollama
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
} if HF_TOKEN else {}


def _safe_offline_answer(context_chunks):
    if not context_chunks:
        return "Information not available."

    top_chunk = context_chunks[0].strip()

    if not top_chunk:
        return "Information not available."

    if "Answer:" in top_chunk:
        return top_chunk.split("Answer:", 1)[1].strip()

    return top_chunk


def _try_huggingface_online(prompt, retries=2):
    if not HF_TOKEN:
        return None, "HF_TOKEN not found"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 80,
            "temperature": 0.2,
            "return_full_text": False
        }
    }

    for attempt in range(retries + 1):
        try:
            print("Using model:", HF_API_URL)

            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=20
            )

            print("HF status:", response.status_code)
            print("HF raw response:", response.text[:300])

            if response.status_code == 503:
                wait_time = 5
                try:
                    result = response.json()
                    if isinstance(result, dict) and "estimated_time" in result:
                        wait_time = max(2, int(result["estimated_time"]))
                except Exception:
                    pass

                if attempt < retries:
                    print(f"Model loading... retrying in {wait_time} seconds")
                    time.sleep(wait_time)
                    continue

                return None, "Hugging Face model is still loading"

            if response.status_code == 401:
                return None, "Invalid or expired HF token"

            if response.status_code == 404:
                return None, "HF model endpoint not found"

            if response.status_code != 200:
                return None, f"HF API error {response.status_code}"

            try:
                result = response.json()
            except Exception:
                return None, "Online model returned non-JSON response"

            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    answer = result[0]["generated_text"].strip()
                    if answer:
                        return answer, None

            if isinstance(result, dict):
                if "generated_text" in result:
                    answer = result["generated_text"].strip()
                    if answer:
                        return answer, None

                if "error" in result:
                    return None, result["error"]

            return None, "Unexpected Hugging Face response format"

        except requests.exceptions.Timeout:
            if attempt < retries:
                print("HF timeout... retrying")
                time.sleep(3)
                continue
            return None, "Online model timeout"

        except Exception as e:
            return None, f"Online model failed: {e}"

    return None, "Unknown online failure"


def _try_ollama_offline(prompt, context_chunks):
    try:
        response = ollama.chat(
            model="tinyllama",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response["message"]["content"].strip()

        if answer:
            return answer

    except Exception as e:
        print("Offline Ollama failed:", e)

    return _safe_offline_answer(context_chunks)


def generate_answer(query, context_chunks):
    if not context_chunks:
        return "Information not available."

    context = "\n\n".join(context_chunks).strip()

    if not context:
        return "Information not available."

    prompt = f"""Answer the question using ONLY the given context.

If the answer is not clearly present, reply exactly:
Information not available.

Context:
{context}

Question:
{query}

Answer:"""

    online_answer, online_error = _try_huggingface_online(prompt)

    if online_answer:
        return online_answer.strip()

    print("Online failed -> switching to offline")
    print("Reason:", online_error)

    return _try_ollama_offline(prompt, context_chunks)