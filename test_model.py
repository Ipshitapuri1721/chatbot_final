import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "./tinyllama-college-bot"

print("🚀 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("🚀 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("🚀 Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_PATH)

model.eval()

print("\n✅ Model ready! Ask questions.\n")

while True:
    question = input("Ask a question: ")

    if question.lower() in ["exit", "quit"]:
        break

    prompt = f"""### Instruction:
{question}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.1,        # stable answers
            do_sample=False,        # deterministic generation
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only the answer
    response = response.split("### Response:")[-1].strip()

    # Remove chat tokens if generated
    response = response.replace("<|user|>", "")
    response = response.replace("<|assistant|>", "")

    print("\nModel Response:")
    print(response)
    print()