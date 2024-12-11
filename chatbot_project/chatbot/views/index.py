from django.shortcuts import render
from django.http import JsonResponse
from project_ecommerce_AI import chatbot_response

from ..models.product import Product
from ..models.models import models

from ..models.menu_options_extended import MenuOptionExtended
from random import sample
from django.views.decorators.csrf import csrf_exempt

# DON'T USE THAT IN PROD
@csrf_exempt
def index(request):
    
    MenuOptionExtended.createMenusOptions()
    all_options = list(MenuOptionExtended.get_all_menuOptions())
    total_option = len(all_options)
    
    # menu_options = all_options[start:end] if start < total_option else []
    
    
    if request.method == 'POST':
        selected_option_id = request.POST.get('selected_option_id')
        
        # Fetch all menu options from the database
        if selected_option_id:
            try:
                # Get the sequence from the POST request
                selected_option_id = MenuOptionExtended.objects.get(menu_extended_id=selected_option_id)
                # Get the related options for the selected menu option
                next_options = selected_option_id.next_options.all()
                
                if not next_options:
                    next_options = False
                
            except MenuOptionExtended.DoesNotExist:
                return JsonResponse({'error': 'The selected option is not selected'}, status=400)

            if next_options:
                return JsonResponse([
                    {'menu_extended_id': option.menu_extended_id, 'menu_option_description': option.menu_option_description} for option in next_options], safe=False)
            else:
                return JsonResponse({'data': 'No more options available'}, safe=False)
        
        user_input = request.POST.get('user_input')
        if user_input:
            response = chatbot_response(user_input)
            return JsonResponse({'response': response})
    
                
        # {'menu_option_description': option.menu_option_description} for option in menu_options], safe=False)
    all_options = all_options[:4]
    return JsonResponse([
        {'menu_extended_id': option.menu_extended_id, 'menu_option_description': option.menu_option_description} for option in all_options
         ], safe=False)
    
    # return render(request, 'index.html', context)
