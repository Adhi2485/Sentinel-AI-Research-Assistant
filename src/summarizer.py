from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class Summarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """
        Generates a summary of the provided text.
        Handles text length constraints (BART limits).
        """
        # Truncate text if it's too long for the model
        # Roughly 1024 tokens max. We'll approximate with words.
        words = text.split()
        if len(words) > 800:
            text = " ".join(words[:800])
            
        if len(words) < 50:
            return text # Too short to summarize
            
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_length, min_length=min_length, do_sample=False)
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()
