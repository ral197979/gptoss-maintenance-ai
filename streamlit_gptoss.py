
# streamlit_gptoss.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

st.title("🧠 Predictive Maintenance AI")
st.markdown("Enter sensor readings and get AI maintenance suggestions.")

BASE_MODEL = "openai/gpt-oss-20b"
LORA_PATH = "./lora_gptoss_maintenance"

@st.cache_resource
def load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(model, LORA_PATH)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

pipe = load_pipeline()
user_input = st.text_area("🔧 Sensor Data Prompt:", height=150)

if st.button("Get Maintenance Insight"):
    if user_input.strip():
        result = pipe(user_input, max_new_tokens=150)[0]["generated_text"]
        st.success("🛠️ AI Suggestion:

" + result)
    else:
        st.warning("Please enter sensor input.")
