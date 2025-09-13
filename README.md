# Text Generation API

A simple FastAPI service that generates text using Hugging Face language models.

## Local setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Test
```bash
curl -X POST "http://127.0.0.1:8000/generate" -H "Content-Type: application/json" -d '{"prompt":"The weather today is", "max_length": 30}'
```

Example prompts:
- `"Once upon a time"`
- `"The benefits of exercise are"`
- `"In the future, technology will"`
- `"My favorite recipe is"`

## Deploy on Render
- Push repo with Dockerfile
- Create new Render Web Service
- You do NOT need to set a fixed port. Render injects $PORT and `start.sh` binds to it.
- Optionally set env `MODEL_ID` to choose a different Hugging Face model (default: `sshleifer/tiny-gpt2`).
- Deploy!

### Configure a different model

Set the environment variable `MODEL_ID` to any Hugging Face repo id, for example:

```bash
MODEL_ID=distilgpt2
MODEL_ID=gpt2
MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Note: larger models take longer to download and may exceed free tier limits.
