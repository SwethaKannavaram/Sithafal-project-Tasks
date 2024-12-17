import openai

def query_with_context(prompt, retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
    
    response = openai.Completion.create(
        model="text-davinci-003",  # Use GPT-4 if available
        prompt=full_prompt,
        max_tokens=500
    )
    return response["choices"][0]["text"].strip()

# Example usage
if __name__ == "__main__":
    openai.api_key = "your-openai-api-key"  # Replace with your API key
    chunks = ["GDP in 2015 was $31 trillion.", "Manufacturing made up 19% of GDP."]
    prompt = "What was the total GDP in 2015?"
    response = query_with_context(prompt, chunks)
    print("Response:", response)
