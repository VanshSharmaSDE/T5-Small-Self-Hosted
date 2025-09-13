from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_manager import ModelManager

app = FastAPI(title="Title → Description API", version="1.0")

model_manager = ModelManager(model_name_or_path="t5-small")

class GenerateRequest(BaseModel):
    title: str
    max_length: int = 60
    num_beams: int = 1

@app.post("/generate", summary="Generate description from title")
async def generate(req: GenerateRequest):
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title must be non-empty.")
    prompt = f"generate description: {title}"
    try:
        desc = model_manager.generate(
            prompt,
            max_length=req.max_length,
            num_beams=req.num_beams
        )
        return {"title": title, "description": desc}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}