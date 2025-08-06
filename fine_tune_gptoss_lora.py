
# LoRA Fine-Tuning Script for Predictive Maintenance using GPT-OSS
# Prerequisites: pip install transformers datasets accelerate peft bitsandbytes

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import torch

# Load GPT-OSS base model (e.g., 20B)
MODEL_NAME = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)

# Load your custom dataset (replace with your actual path or Hugging Face dataset)
dataset = load_dataset("json", data_files={"train": "train_data.json", "test": "test_data.json"})

# Tokenize dataset
def tokenize(example):
    inputs = tokenizer(example["prompt"], padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(example["response"], padding="max_length", truncation=True, max_length=512)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_datasets = dataset.map(tokenize, batched=True)

# Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, peft_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora_gptoss_maintenance",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    evaluation_strategy="epoch",
    logging_steps=10,
    num_train_epochs=3,
    fp16=True,
    save_strategy="epoch",
    learning_rate=2e-4,
    warmup_steps=20,
    save_total_limit=2,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Start training
trainer.train()
