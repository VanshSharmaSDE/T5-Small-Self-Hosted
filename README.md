# Title → Description API (t5-small)

## Local setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Test
```bash
curl -X POST "http://127.0.0.1:8000/generate" -H "Content-Type: application/json" -d '{"title":"Smart Water Bottle"}'
```

## Deploy on Render
- Push repo with Dockerfile
- Create new Render Web Service
- Set port to 8000
- Deploy!
