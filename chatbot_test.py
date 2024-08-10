import google.generativeai as genai

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

if __name__ == "__main__":
    # generate_response_stream_chat()
    
    generate_response_interactive_chat()
    
    