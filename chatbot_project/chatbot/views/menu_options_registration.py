from django.shortcuts import render, redirect

from django.contrib.auth.hashers import check_password
from django.views import View

from ..models.menu_options import *
from django.http import JsonResponse

from random import sample

def is_ajax(request):
    return request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'

class MenuOptionRegistration(View):
    
    def get(self, request):
        query = request.GET.get('search', '') # Get the search term from the request
        
        if is_ajax(request) and query:
            menuOptions = MenuOption.objects.filter(productName__icontains=query) # Query the database
            
            menuOption_name = list(menuOptions.values('menuOptionName', flat=True).distinct()) # Get a list of product names
            return JsonResponse(menuOption_name, safe=False)
        
        return render(request, 'register/chatbot_menu_options.html')
    
    
    
    def post(self, request):
        menuOptionName = request.POST.get('menuOptionNameInput')
        menuOptionDescription = request.POST.get('menuOptionDescriptionInput')
        
        menuOptions = MenuOption.get_all_menuOptions()
        
                
        menuOption = MenuOption(menu_option_name=menuOptionName, menu_option_description=menuOptionDescription)
        
        
        menuOption.save()
        
        MenuOption.create_menu_options_prompts()
           
        return redirect('index')
    
    