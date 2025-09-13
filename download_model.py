from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

MODEL_NAME = "distilgpt2"
TARGET_DIR = os.path.join("models", MODEL_NAME)

os.makedirs(TARGET_DIR, exist_ok=True)
print(f"Downloading {MODEL_NAME} to {TARGET_DIR} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(TARGET_DIR)
model.save_pretrained(TARGET_DIR)
print("Done.")