from sentence_transformers import SentenceTransformer

def chunk_text(text_data, chunk_size=500):
    chunks = []
    for entry in text_data:
        text = entry["content"]
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
    return chunks

def create_embeddings(text_chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_chunks)
    return embeddings, model

if __name__ == "__main__":
    text_data = [{"content": "This is a test chunk of text for embedding."}]
    chunks = chunk_text(text_data)
    embeddings, model = create_embeddings(chunks)
    print("Chunks:", chunks)
    print("Embeddings Shape:", embeddings.shape)
