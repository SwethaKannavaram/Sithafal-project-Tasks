from pdf_extractor import extract_text_and_tables
from embedding_creator import chunk_text, create_embeddings
from faiss_index import create_faiss_index, search_index
from llm_responder import query_with_context

def rag_pipeline(pdf_path, query):
    text_data, tables = extract_text_and_tables(pdf_path)
    chunks = chunk_text([entry["content"] for entry in text_data])
    embeddings, embedding_model = create_embeddings(chunks)
    index = create_faiss_index(embeddings)
    retrieved_chunks = search_index(query, index, embedding_model, chunks)
    response = query_with_context(query, retrieved_chunks)
    return response

if __name__ == "__main__":
    pdf_path = "table.pdf"
    query = "What was the GDP of all industries in 2015?"
    response = rag_pipeline(https://www.hunter.cuny.edu/dolciani/pdf_files/workshop-materials/mmc-presentations/tables-charts-and-graphs-with-examples-from.pdf , query)
    print("Final Response:", response)
