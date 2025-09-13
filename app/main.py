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
import gc

# ---- Reduce transformers logging noise (suppresses the legacy-info line) ----
hf_logging.set_verbosity_error()

# ---- Config ----
# Model size options (from smallest to largest):
# - "prajjwal1/t5-tiny" (smallest, ~17MB)
# - "google/t5-efficient-tiny" (small and efficient)
# - "google/t5-efficient-mini" (medium size)
# - "t5-small" (original, larger)
MODEL_NAME = os.getenv("MODEL_NAME", "prajjwal1/t5-tiny")  # Much smaller than t5-small
MODEL_SIZE = os.getenv("MODEL_SIZE", "tiny")  # Options: tiny, efficient-tiny, efficient-mini, small

# Map MODEL_SIZE to actual model name if MODEL_NAME is not explicitly set
if "MODEL_NAME" not in os.environ and MODEL_SIZE:
    size_to_model = {
        "tiny": "prajjwal1/t5-tiny",
        "efficient-tiny": "google/t5-efficient-tiny",
        "efficient-mini": "google/t5-efficient-mini",
        "small": "t5-small"
    }
    MODEL_NAME = size_to_model.get(MODEL_SIZE, "prajjwal1/t5-tiny")

PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", "1")  # "1" loads model in background at startup
KEEP_ALIVE_URL = os.getenv("RENDER_URL")        # e.g. https://t5-small-self-hosted.onrender.com
KEEP_ALIVE_INTERVAL = int(os.getenv("KEEP_ALIVE_INTERVAL", "600"))  # seconds
USE_INT8 = os.getenv("USE_INT8", "1") == "1"    # Use 8-bit quantization to reduce memory usage

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
        
        # Force garbage collection before loading model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logging.info("Loading model %s ...", MODEL_NAME)
        
        # Check if model exists locally first (from download_model.py)
        local_model_dir = os.path.join("models", MODEL_NAME.replace("/", "_"))
        model_exists_locally = os.path.exists(local_model_dir)
        
        if model_exists_locally:
            logging.info(f"Loading model from local directory: {local_model_dir}")
            tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Load model with reduced memory footprint
        if USE_INT8:
            try:
                # Try to use 8-bit quantization if available
                from transformers import AutoModelForSeq2SeqLM
                import bitsandbytes as bnb
                
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    local_model_dir if model_exists_locally else MODEL_NAME,
                    load_in_8bit=True,
                    device_map="auto",
                )
                logging.info("Model loaded with 8-bit quantization")
            except (ImportError, Exception) as e:
                logging.warning(f"8-bit quantization failed: {e}. Using standard model loading.")
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    local_model_dir if model_exists_locally else MODEL_NAME, 
                    low_cpu_mem_usage=True
                )
        else:
            # Standard loading with memory optimization
            model = AutoModelForSeq2SeqLM.from_pretrained(
                local_model_dir if model_exists_locally else MODEL_NAME,
                low_cpu_mem_usage=True
            )
        
        model.eval()
        
        # Only move to GPU if available AND not using 8-bit quantization (which already handles device placement)
        if torch.cuda.is_available() and not USE_INT8:
            model.to("cuda")
            
        _model_ready = True
        logging.info("Model loaded successfully.")
        
        # Report memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            logging.info(f"GPU memory allocated: {allocated:.2f} MB")
        else:
            import psutil
            process = psutil.Process(os.getpid())
            memory = process.memory_info().rss / (1024 * 1024)
            logging.info(f"RAM memory used: {memory:.2f} MB")

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

# Memory usage monitoring endpoint
@app.get("/memory")
def memory_usage():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        gpu_info = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info[f"gpu_{i}_allocated"] = f"{torch.cuda.memory_allocated(i) / (1024**2):.2f} MB"
                gpu_info[f"gpu_{i}_reserved"] = f"{torch.cuda.memory_reserved(i) / (1024**2):.2f} MB"
        
        return {
            "model": MODEL_NAME,
            "ram_usage_mb": f"{memory_mb:.2f}",
            "percent_ram": f"{process.memory_percent():.2f}%",
            "gpu": gpu_info,
            "model_loaded": _model_ready,
            "quantization": "8-bit" if USE_INT8 else "default"
        }
    except Exception as e:
        return {"error": str(e)}

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

    try:
        # Force garbage collection before inference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        prompt = f"generate description: {title}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        
        # Move inputs to device if needed and not using 8-bit quantization
        if torch.cuda.is_available() and not USE_INT8:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=req.max_length,
                num_beams=req.num_beams,
                early_stopping=True,
            )
        desc = tokenizer.decode(out[0], skip_special_tokens=True)
        
        # Force cleanup after inference
        del inputs, out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return {"title": title, "description": desc}
    except Exception as e:
        logging.exception(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

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
