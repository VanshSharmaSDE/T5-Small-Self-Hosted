import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class ModelManager:
    def __init__(self, model_name_or_path: str = "t5-small", device: str = None):
        if device:
            self.device = device
        else:
            force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
            self.device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
        self.model_name = model_name_or_path
        self.tokenizer = None
        self.model = None
        self._load()

    def _load(self):
        print(f"Loading model {self.model_name} on device {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        if self.device == "cuda":
            self.model = self.model.to("cuda")
        self.model.eval()
        print("Model loaded.")

    def generate(self, prompt: str, max_length: int = 60, num_beams: int = 1) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)