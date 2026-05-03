import json
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.retrieval import RAGPipeline

def evaluate_system():
    print("Initializing RAG Pipeline for Evaluation...")
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Load semantic evaluator model
    print("Loading Semantic Evaluator Model...")
    eval_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    # Define a test set of questions and expected answers
    # Users can modify this based on the PDFs they uploaded!
    test_set = [
        {
            "query": "What are the six types of waste classified in the system?",
            "expected_answer": "The system classifies waste into six categories: glass, paper, cardboard, plastic, metal, and general trash."
        },
        {
            "query": "How does ConvoWaste differ from Capsule-Net?",
            "expected_answer": "ConvoWaste uses a convolutional neural network approach optimized for real-time waste sorting, whereas Capsule-Net utilizes capsule layers to capture hierarchical spatial relationships in images, often requiring more computational power."
        }
    ]
    
    print("\n--- Starting Evaluation ---")
    results = []
    
    for item in test_set:
        query = item["query"]
        expected = item["expected_answer"]
        
        print(f"\nEvaluating Query: '{query}'")
        
        # 1. Run Query through our System
        response = pipeline.query(query, top_k=5)
        generated_answer = response["answer"]
        
        # 2. Calculate ROUGE Scores (Lexical overlap)
        rouge_scores = scorer.score(expected, generated_answer)
        rouge1_fmeasure = rouge_scores['rouge1'].fmeasure
        rougeL_fmeasure = rouge_scores['rougeL'].fmeasure
        
        # 3. Calculate Semantic Similarity (Meaning overlap)
        expected_emb = eval_model.encode([expected])
        generated_emb = eval_model.encode([generated_answer])
        semantic_sim = cosine_similarity(expected_emb, generated_emb)[0][0]
        
        print(f"Generated Answer: {generated_answer}")
        print(f"-> ROUGE-1 F1: {rouge1_fmeasure:.4f}")
        print(f"-> ROUGE-L F1: {rougeL_fmeasure:.4f}")
        print(f"-> Semantic Similarity: {semantic_sim:.4f}")
        
        results.append({
            "query": query,
            "rouge1": rouge1_fmeasure,
            "rougeL": rougeL_fmeasure,
            "semantic_similarity": float(semantic_sim)
        })
        
    # Final Metrics
    print("\n==========================================")
    print("🏆 FINAL SYSTEM METRICS")
    print("==========================================")
    avg_rouge1 = np.mean([r["rouge1"] for r in results])
    avg_rougeL = np.mean([r["rougeL"] for r in results])
    avg_semantic = np.mean([r["semantic_similarity"] for r in results])
    
    print(f"Average ROUGE-1 Score: {avg_rouge1:.4f} (Measures word overlap)")
    print(f"Average ROUGE-L Score: {avg_rougeL:.4f} (Measures sentence structure overlap)")
    print(f"Average Semantic Similarity: {avg_semantic:.4f} (Measures meaning preservation)")
    print("==========================================")
    print("Note: If scores are extremely low, ensure the PDFs containing the answers are currently indexed!")

if __name__ == "__main__":
    evaluate_system()
