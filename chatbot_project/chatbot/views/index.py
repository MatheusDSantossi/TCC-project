from django.shortcuts import render
from django.http import JsonResponse
from project_ecommerce_AI import chatbot_response

from ..models.product import Product
from ..models.models import models

from ..models.menu_options_ordered import MenuOptionOrdered
from random import sample

def index(request):
    
    # Fetch all menu options from the database
    all_options = list(MenuOptionOrdered.get_all_menuOptions())
    
    # Get the first 4 random options
    # menu_options = sample(all_options, 4) if len(all_options) >= 4 else all_options
    
    # Get the first 4 options
    menu_options = all_options[:4]
    
    # context = {
    #     'menu_options': menu_options,
    # }
        
    # if request.method == 'POST':
        
    #     user_input = request.POST.get('user_input')
    #     response = chatbot_response(user_input)
        
                
    return JsonResponse([{'menu_option_description': option.menu_option_description} for option in menu_options], safe=False)
    
    # return render(request, 'index.html', context)
