from django.shortcuts import render
from django.http import JsonResponse
from project_ecommerce_AI import chatbot_response
# from chatbot.models.models import models
# from chatbot.models.product import product
from ..models.product import Product
from ..models.models import models

# from ..models.menu_options import MenuOption
from ..models.menu_options_ordered import MenuOptionOrdered
from random import sample

def index(request):
    
    # Fetch all menu options from the database
    all_options = list(MenuOptionOrdered.get_all_menuOptions())
    
    # Get the first 4 random options
    # menu_options = sample(all_options, 4) if len(all_options) >= 4 else all_options
    
    # Get the first 4 options
    menu_options = all_options[:4]
    
    context = {
        'menu_options': menu_options,
    }
    
    # return render(request, 'chatbot/chatbot_template.html', context)
    
    if request.method == 'POST':
        
        user_input = request.POST.get('user_input')
        response = chatbot_response(user_input)
        
                
        return JsonResponse({'response': response})
    
    # elif request.method == 'GET':
    #     return render(request, 'chatbot/edit_menu_options_list.html', context)
    
    return render(request, 'base/base.html', context)
    # return render(request, 'chatbot/index.html')


# def menu_option_chatbot_view(request):
#     # Fetch all menu options from the database
#     all_options = list(MenuOption.objects.all())
    
#     # Get 4 random options
#     menu_options = sample(all_options, 4) if len(all_options) >= 4 else all_options
    
#     context = {
#         'menu_options': menu_options,
#     }
    
#     return render(request, 'chatbot/chatbot_template.html', context)

