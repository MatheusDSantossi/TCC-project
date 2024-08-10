from django.shortcuts import render
from django.http import JsonResponse
from project_ecommerce_AI import chatbot_response
# from chatbot.models.models import models
# from chatbot.models.product import product
from ..models.product import Product
from ..models.models import models

def index(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        response = chatbot_response(user_input)
        
        
                
        return JsonResponse({'response': response})
    
    return render(request, 'base/base.html')
    # return render(request, 'chatbot/index.html')

# def cadastrar_product(request):
#     product = Product.objects.all()
#     return render(request, 'product.html', {'product': product})
