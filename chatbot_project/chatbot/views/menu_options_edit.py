from django.core.cache import cache

from django.shortcuts import render, redirect

from django.contrib.auth.hashers import check_password
from django.views import View

import json

from ..models.menu_options import *
from ..models.menu_options_ordered import *
from django.http import JsonResponse

from random import sample

def is_ajax(request):
    return request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'

class MenuOptionEdit(View):
    
    def get(self, request):
        # Fetch all menu options from the database
        all_options = list(MenuOption.get_all_menuOptions())
        
        # Get 4 random options
        menu_options = all_options
        
        context = {
            'menu_options': menu_options,
        }
        
        query = request.GET.get('search', '') # Get the search term from the request
        
        if is_ajax(request) and query:
            menuOptions = MenuOption.objects.filter(productName__icontains=query) # Query the database
            
            menuOption_name = list(menuOptions.values('menuOptionName', flat=True).distinct()) # Get a list of product names
            return JsonResponse(menuOption_name, safe=False)
        
        return render(request, 'chatbot/edit_menu_options_list.html', context)
    
    
    
    def post(self, request):
        
        menuOptionDescription = request.POST.get('menuOptionDescriptionInput')
        
        menuOptions = MenuOptionOrdered.get_all_menuOptions()
        
        menuOption = MenuOptionOrdered(menu_option_description=menuOptionDescription)
        
        menuOption.save()
           
        return redirect('index')
    
    def save_menu_order(request):
        if request.method == 'POST':
            # invalidate the cache (if used)
            cache.clear()
            
            try:
                data = json.loads(request.body)
                menu_options = data.get('menuOptions', [])
                
                MenuOptionOrdered.objects.all().delete()
                
                # for description in menu_options[:4]:
                for description in menu_options[:4]:
                    menu_option = MenuOptionOrdered(menu_option_description=description)
                    menu_option.save()
                    
                # Fetcg tge updated menu options to ensure it reflets in the next request
                menu_options = list(MenuOptionOrdered.get_all_menuOptions())
                    
                return JsonResponse({'success': True, 'menuOptions': [opt.menu_option_description for opt in menu_options]})
                
            except Exception as e:
                return JsonResponse({'success': False, 'error': str(e)})
            
        return JsonResponse({'success': False, 'error': 'Invalid request method'})
    
    