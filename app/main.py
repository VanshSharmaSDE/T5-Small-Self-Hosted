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
    
    # Fix pad token issue - set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_loaded = True
    print(f"Model and tokenizer loaded successfully from '{MODEL_ID}'!")
except Exception as e:
    model_error = str(e)
    print(f"Error loading model '{MODEL_ID}': {model_error}")

app = FastAPI(title="Text Generation API")

class TextInput(BaseModel):
    prompt: str
    max_length: int = 50

@app.post("/generate")
def generate_text(data: TextInput):
    # Check if model is loaded
    if not model_loaded:
        raise HTTPException(
            status_code=503, 
            detail=f"Model not available: {model_error}"
        )
    
    # Use the prompt directly without modification
    prompt = data.prompt.strip()
    
    # Encode with attention mask
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=min(inputs.shape[1] + data.max_length, 100),  # Limit total length
        do_sample=True,
        temperature=0.8,  # More creative
        top_p=0.9,
        top_k=50,  # Add top-k sampling for better quality
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.1,  # Reduce repetition
        early_stopping=True  # Stop when EOS token is generated
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the original prompt from the output
    if prompt in generated_text:
        completion = generated_text[len(prompt):].strip()
    else:
        completion = generated_text.strip()
    
    return {
        "prompt": data.prompt,
        "completion": completion,
        "full_text": generated_text
    }

@app.get("/")
def root():
    if model_loaded:
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_id": MODEL_ID,
            "message": "Language model is ready to generate text"
        }
    else:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "model_id": MODEL_ID,
            "error": model_error,
            "message": "Model failed to load"
        }
