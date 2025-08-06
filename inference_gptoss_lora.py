
# inference_gptoss_lora.py
# Run your fine-tuned GPT-OSS LoRA model in a CLI script

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from peft import PeftModel

# Load base model and LoRA adapter
BASE_MODEL = "openai/gpt-oss-20b"
LORA_PATH = "./lora_gptoss_maintenance"  # directory with adapter config

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, LORA_PATH)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

while True:
    prompt = input("🔧 Enter sensor data (or 'exit'): ")
    if prompt.lower() == "exit":
        break
    result = pipe(prompt, max_new_tokens=150)[0]['generated_text']
    print("🧠 AI Maintenance Insight:
", result)
