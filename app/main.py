# app/main.py
import os
import threading
import time
import logging
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging as hf_logging
import torch

# ---- Reduce transformers logging noise (suppresses the legacy-info line) ----
hf_logging.set_verbosity_error()

# ---- Config ----
MODEL_NAME = os.getenv("MODEL_NAME", "t5-small")
PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", "1")  # "1" loads model in background at startup
KEEP_ALIVE_URL = os.getenv("RENDER_URL")        # e.g. https://t5-small-self-hosted.onrender.com
KEEP_ALIVE_INTERVAL = int(os.getenv("KEEP_ALIVE_INTERVAL", "600"))  # seconds

app = FastAPI(title="T5 Title→Description")

# ---- Model globals ----
tokenizer = None
model = None
_model_lock = threading.Lock()
_model_ready = False

def load_model():
    """Load tokenizer + model. Safe to call multiple times (uses lock)."""
    global tokenizer, model, _model_ready
    with _model_lock:
        if _model_ready:
            return
        logging.info("Loading model %s ...", MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model.eval()
        # move to GPU if available (optional)
        if torch.cuda.is_available():
            model.to("cuda")
        _model_ready = True
        logging.info("Model loaded.")

def _background_model_loader():
    try:
        load_model()
    except Exception:
        logging.exception("Background model load failed")

# Start background model load if configured (non-blocking)
if PRELOAD_MODEL != "0":
    t = threading.Thread(target=_background_model_loader, daemon=True)
    t.start()

# ---- API types ----
class GenerateRequest(BaseModel):
    title: str
    max_length: int = 60
    num_beams: int = 1

# Lightweight health endpoint used by Render and keep-alive
@app.get("/health")
def health():
    status = "ready" if _model_ready else "loading"
    return {"status": status}

# Main generation endpoint
@app.post("/generate")
def generate(req: GenerateRequest):
    global _model_ready
    title = (req.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title must be non-empty.")

    # If model not ready, wait up to WAIT_SECONDS then fail with 503
    if not _model_ready:
        WAIT_SECONDS = 30
        waited = 0
        while waited < WAIT_SECONDS and not _model_ready:
            time.sleep(1)
            waited += 1
        if not _model_ready:
            raise HTTPException(status_code=503, detail="Model is loading; try again shortly.")

    prompt = f"generate description: {title}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_length=req.max_length,
            num_beams=req.num_beams,
            early_stopping=True,
        )
    desc = tokenizer.decode(out[0], skip_special_tokens=True)
    return {"title": title, "description": desc}

# ---- Optional keep-alive ping (self-ping) ----
def _keep_alive_worker():
    if not KEEP_ALIVE_URL:
        return
    url = KEEP_ALIVE_URL.rstrip("/") + "/health"
    while True:
        try:
            requests.get(url, timeout=10)
        except Exception as e:
            logging.debug("Keep-alive ping failed: %s", e)
        time.sleep(KEEP_ALIVE_INTERVAL)

if KEEP_ALIVE_URL:
    t2 = threading.Thread(target=_keep_alive_worker, daemon=True)
    t2.start()
