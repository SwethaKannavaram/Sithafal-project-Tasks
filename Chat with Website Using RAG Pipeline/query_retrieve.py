import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load FAISS index
index = faiss.read_index("website_embeddings.index")

# Load the document data
with open("documents.txt", "r") as f:
    documents = f.readlines()

# Function to convert text content to vector embeddings
def get_embeddings(text, model):
    embeddings = model.encode([text], show_progress_bar=False)
    return embeddings

# Function to query the database and retrieve relevant chunks
def query_database(query, model, index, top_k=5):
    query_embedding = get_embeddings(query, model)
    _, indices = index.search(query_embedding, top_k)
    return indices

# Load the SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# User query
user_query = "What is the mission of Stanford University?"

# Query the database
relevant_indices = query_database(user_query, embedding_model, index)

# Retrieve the most relevant documents
relevant_docs = [documents[i].strip() for i in relevant_indices[0]]

# Print the retrieved documents
print("Retrieved Documents:")
for doc in relevant_docs:
    print(doc)
