# app.py
import gradio as gr
from transformers import pipeline

pipe = pipeline("text-generation", model="openai/gpt-oss-20b", device_map="auto")

def chat(prompt):
    result = pipe(prompt, max_new_tokens=150)[0]["generated_text"]
    return result

demo = gr.Interface(fn=chat, inputs="text", outputs="text", title="GPT-OSS Demo")
demo.launch()
