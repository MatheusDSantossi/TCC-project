import requests

def search_image(product_name):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'product_name': product_name,
        'search_engine': "chatbotImageSearch",
        'api_key': "AIzaSyCcK7v5ws7aK3rc4_Gknv_Ar6F-eECaQPw",
        'serachType': "image",
        'num': 1,
        }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'items' in data:
        return data['items'][0]['link']
    
    return None
