import faiss

def create_faiss_index(embeddings):
    d = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def search_index(query, index, embedding_model, chunks, top_k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

# Example usage
if __name__ == "__main__":
    import numpy as np
    from sentence_transformers import SentenceTransformer
    
    # Sample embeddings and query
    chunks = ["This is chunk 1.", "This is chunk 2."]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    index = create_faiss_index(embeddings)
    
    query = "chunk 1"
    results = search_index(query, index, model, chunks)
    print("Search Results:", results)
