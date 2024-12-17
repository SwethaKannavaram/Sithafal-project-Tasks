import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Function to scrape a website
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extracting relevant content (headers, paragraphs, etc.)
    content = ""
    for paragraph in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        content += paragraph.get_text() + "\n"

    return content

# Function to convert text content to vector embeddings
def get_embeddings(text, model):
    embeddings = model.encode([text], show_progress_bar=True)
    return embeddings

# Setup SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Example URLs to scrape
urls = [
    "https://www.uchicago.edu/",
    "https://www.washington.edu/",
    "https://www.stanford.edu/"
]

# Scrape and process each website
documents = []
for url in urls:
    text_content = scrape_website(url)
    documents.append(text_content)

# Convert text content into embeddings
embeddings = np.vstack([get_embeddings(doc, embedding_model) for doc in documents])

# Storing embeddings in FAISS
dimension = embeddings.shape[1]  # Embedding dimension
index = faiss.IndexFlatL2(dimension)  # Create a FAISS index
index.add(embeddings)  # Add embeddings to the index

# Save the index to disk (optional)
faiss.write_index(index, "website_embeddings.index")
 