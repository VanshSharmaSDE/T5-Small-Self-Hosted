from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Hugging Face model id (configurable via env var MODEL_ID)
# Default uses a tiny GPT-2 for fast startup
MODEL_ID = os.getenv("MODEL_ID", "sshleifer/tiny-gpt2")

# Model loading status
model_loaded = False
model_error = None
tokenizer = None
model = None

# Load tokenizer & model from Hugging Face with error handling
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model_loaded = True
    print(f"Model and tokenizer loaded successfully from '{MODEL_ID}'!")
except Exception as e:
    model_error = str(e)
    print(f"Error loading model '{MODEL_ID}': {model_error}")

app = FastAPI(title="Tiny GPT-2 API")

class TitleIn(BaseModel):
    title: str

@app.post("/generate")
def generate_description(data: TitleIn):
    # Check if model is loaded
    if not model_loaded:
        raise HTTPException(
            status_code=503, 
            detail=f"Model not available: {model_error}"
        )
    
    # Create a prompt for description generation
    prompt = f"Title: {data.title}\nDescription:"
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs,
        max_length=100,  # Increased for better descriptions
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the description part (after "Description:")
    if "Description:" in generated_text:
        description = generated_text.split("Description:")[-1].strip()
    else:
        description = generated_text.replace(prompt, "").strip()
    
    return {
        "title": data.title,
        "description": description
    }

@app.get("/")
def root():
    if model_loaded:
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_id": MODEL_ID,
            "message": "Language model is ready to generate descriptions"
        }
    else:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "model_id": MODEL_ID,
            "error": model_error,
            "message": "Model failed to load"
        }
