from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class QAModel:
    def __init__(self, model_name: str = "google/flan-t5-large"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
    def generate_answer(self, query: str, context_chunks: list[dict]) -> str:
        """
        Generates an answer from the retrieved context.
        """
        if not context_chunks:
            return "I don't have enough information to answer that question."
            
        # Combine text from chunks
        raw_context = " ".join([c["chunk"]["text"] for c in context_chunks])
        
        # System instructions for a highly professional output
        instruction = "You are an expert AI research assistant. Provide a comprehensive, detailed, and highly professional answer to the question using ONLY the provided context. If the context does not contain the answer, say so explicitly. Do not make up information."
        
        # Build prompt skeleton
        prompt_skeleton = f"{instruction}\n\nContext:\n{{}}\n\nQuestion: {query}\n\nComprehensive Answer:"
        
        # Calculate how many tokens we can allocate to the context
        skeleton_tokens = len(self.tokenizer.encode(prompt_skeleton.format("")))
        max_context_tokens = 512 - skeleton_tokens - 10 # 512 is safe limit for T5
        
        if max_context_tokens < 0:
            max_context_tokens = 100 # Fallback
            
        # Truncate context by tokens
        context_tokens = self.tokenizer.encode(raw_context, truncation=True, max_length=max_context_tokens)
        context = self.tokenizer.decode(context_tokens, skip_special_tokens=True)
        
        # Final prompt
        prompt = prompt_skeleton.format(context)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generation with beam search and repetition penalty for higher quality
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=350, 
            min_length=20,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
