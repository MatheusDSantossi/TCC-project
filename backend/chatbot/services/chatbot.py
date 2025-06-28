from .product_ai import chatbot_response

def generate_chatbot_response(user_iput: str, context: str, business_id: int):
    # any pre- or post-processing can live here
    return chatbot_response(user_iput, context, business_id)
