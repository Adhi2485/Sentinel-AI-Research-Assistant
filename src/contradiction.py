from transformers import pipeline
import torch

class ContradictionDetector:
    def __init__(self, model_name: str = "cross-encoder/nli-distilroberta-base"):
        device = 0 if torch.cuda.is_available() else -1
        # NLI models usually output classification: contradiction, entailment, neutral
        self.classifier = pipeline("text-classification", model=model_name, device=device)
        
    def detect(self, premise: str, hypothesis: str) -> dict:
        """
        Compares two text strings to determine if the hypothesis contradicts the premise.
        """
        # Formulate as premise + hypothesis pair for the cross-encoder
        text_pair = f"{premise} </s></s> {hypothesis}"
        try:
            result = self.classifier(text_pair)[0]
            # Mapping depends on the model, cross-encoder/nli-distilroberta-base:
            # LABEL_0: contradiction, LABEL_1: entailment, LABEL_2: neutral
            label_map = {
                "LABEL_0": "contradiction",
                "LABEL_1": "entailment",
                "LABEL_2": "neutral",
                "contradiction": "contradiction",
                "entailment": "entailment",
                "neutral": "neutral"
            }
            label = label_map.get(result["label"].lower(), result["label"].lower())
            
            return {
                "label": label,
                "score": result["score"]
            }
        except Exception as e:
            return {"label": "error", "score": 0.0, "details": str(e)}

    def analyze_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Analyzes pairs of chunks to find potential contradictions.
        Only compares chunks from different documents if available, or just pairs.
        """
        contradictions = []
        n = len(chunks)
        
        for i in range(n):
            for j in range(i + 1, n):
                chunk1 = chunks[i]["chunk"]
                chunk2 = chunks[j]["chunk"]
                
                # Check if chunk2 contradicts chunk1
                res1 = self.detect(chunk1["text"], chunk2["text"])
                if res1["label"] == "contradiction" and res1["score"] > 0.6:
                    contradictions.append({
                        "source_1": f"[{chunk1['document_name']}, Page {chunk1['page_number']}]",
                        "text_1": chunk1["text"],
                        "source_2": f"[{chunk2['document_name']}, Page {chunk2['page_number']}]",
                        "text_2": chunk2["text"],
                        "score": res1["score"]
                    })
                    
        return contradictions
