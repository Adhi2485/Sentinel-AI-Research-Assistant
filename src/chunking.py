def chunk_text(pages_data: list[dict], chunk_size: int = 250, overlap: int = 40) -> list[dict]:
    """
    Splits page text into overlapping word chunks.
    Preserves metadata (document name, page number) for each chunk.
    """
    chunks = []
    
    for page in pages_data:
        text = page["text"]
        words = text.split()
        
        if not words:
            continue
            
        start_idx = 0
        while start_idx < len(words):
            end_idx = min(start_idx + chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "document_name": page["document_name"],
                "page_number": page["page_number"],
                "text": chunk_text
            })
            
            start_idx += (chunk_size - overlap)
            
    return chunks
