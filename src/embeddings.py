from sentence_transformers import SentenceTransformer
import torch

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        
    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Converts a list of texts into dense vector representations.
        """
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
