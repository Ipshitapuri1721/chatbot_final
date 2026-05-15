import os
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# -------------------------------------------------
# Force CUDA
# -------------------------------------------------
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

print("🚀 Using CUDA GPU")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = os.path.join("data", "train.json")
OUTPUT_DIR = "./tinyllama-college-bot"

NUM_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 2e-4

# -------------------------------------------------
# Dataset
# -------------------------------------------------
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def format_example(example):
    return {
        "text": f"### Instruction:\n{str(example['instruction']).strip()}\n\n### Response:\n{str(example['output']).strip()}"
    }

dataset = dataset.map(
    format_example,
    remove_columns=dataset.column_names
)

# -------------------------------------------------
# Tokenizer
# -------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------------------------------------
# 4-bit Model
# -------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
)

# -------------------------------------------------
# LoRA
# -------------------------------------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# -------------------------------------------------
# Training Args (NO AMP ANYWHERE)
# -------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    logging_steps=10,
    save_strategy="epoch",

    fp16=False,
    bf16=False,
    optim="adamw_torch",
    max_grad_norm=0.0,
    report_to="none",
)

# -------------------------------------------------
# Trainer
# -------------------------------------------------
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)

trainer.train()

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training complete")