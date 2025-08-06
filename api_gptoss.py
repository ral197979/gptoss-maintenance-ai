
# api_gptoss.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

app = FastAPI()

BASE_MODEL = "openai/gpt-oss-20b"
LORA_PATH = "./lora_gptoss_maintenance"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, LORA_PATH)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

class PromptInput(BaseModel):
    prompt: str

@app.post("/infer")
def infer(input: PromptInput):
    output = pipe(input.prompt, max_new_tokens=150)[0]['generated_text']
    return {"response": output}
