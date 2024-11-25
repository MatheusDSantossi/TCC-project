import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

genai.configure(api_key="AIzaSyBskS4hcTWUP0L0QROEbuPEKdvp4KVlWkQ")

def generate_response_stream_chat():

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("Write a story about a magic backpack.", stream=True)

    for chunk in response:
        print(chunk.text)
        print("_" * 80)

def generate_response_interactive_chat():
    model = genai.GenerativeModel("gemini-1.5-flash")
    chat = model.start_chat(
        history=[
            {"role": "user", "parts": "Hello"},
            {"role": "model", "parts": "Great to meet you. What would you like to know?"},
        ]
    )
    
    response = chat.send_message("I have a supermarket called Santos")
    print(response.text)
    print("-"*80)
    response = chat.send_message("What's the name of my supermarmekt?")
    print(response.text)


def embeddings_testing(input_embed, embeddings):
    
   
                
    # similarities = cosine_similarity([input_embed], embeddings)[0]
    similarities = cosine_similarity([input_embedding], embeddings)[0]
    
    # Find the most similar embeddings and their indices
    most_similar_indices = np.argsort(similarities)[::-1][:20]
    
    return most_similar_indices

if __name__ == "__main__":
    # generate_response_stream_chat()
    print("TESTING EMBEDDING: ", embeddings_testing())
    # generate_response_interactive_chat()
    
    