from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the generation model (e.g., T5 or GPT-like model)
generation_model_name = "t5-small"  # Replace with your preferred model
generation_model = AutoModelForSeq2SeqLM.from_pretrained(generation_model_name)
generation_tokenizer = AutoTokenizer.from_pretrained(generation_model_name)

# Function to generate responses using a language model
def generate_response(query, retrieved_docs, model, tokenizer):
    # Combine query and retrieved documents as input for the generation model
    input_text = f"Query: {query}\\nRelevant Documents: {retrieved_docs}\\nAnswer:"
    
    # Tokenize input
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate output
    outputs = model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
    
    # Decode and return the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Load retrieved documents
with open("retrieved_docs.txt", "r") as f:
    retrieved_docs = f.read()

# User query
user_query = "What is the mission of Stanford University?"

# Generate a response
response = generate_response(user_query, retrieved_docs, generation_model, generation_tokenizer)

# Print the response
print("Generated Response:")
print(response)

# Optional: Save the query, relevant documents, and response for future reference
output_log = {
    "query": user_query,
    "retrieved_docs": retrieved_docs,
    "response": response
}

# Save to a file
output_file = "query_response_log.json"
import json
with open(output_file, "w") as f:
    json.dump(output_log, f, indent=4)

print(f"Query and response saved to {output_file}")
