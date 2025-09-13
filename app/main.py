from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import threading
import time
import requests
import os

app = Flask(__name__)

# Load model + tokenizer once
MODEL_NAME = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "T5 Small Description Generator is running!"})

@app.route("/generate", methods=["POST"])
def generate_description():
    data = request.get_json()
    title = data.get("title", "")

    if not title:
        return jsonify({"error": "Title is required"}), 400

    input_text = f"generate description: {title}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=64, truncation=True)
    outputs = model.generate(inputs, max_length=100, num_beams=4, early_stopping=True)
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"title": title, "description": description})

@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}

# --- 🔥 Keep Alive Function ---
def keep_alive():
    url = os.environ.get("RENDER_URL", "https://t5-small-tickr.onrender.com")
    while True:
        try:
            requests.get(url, timeout=10)
            print(f"Pinged {url} to keep alive.")
        except Exception as e:
            print(f"Keep-alive failed: {e}")
        time.sleep(600)  # every 10 minutes


# Start keep-alive thread when app starts
threading.Thread(target=keep_alive, daemon=True).start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
