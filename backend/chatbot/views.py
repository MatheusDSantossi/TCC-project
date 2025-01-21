from django.shortcuts import render
from django.http import JsonResponse
from utils.project_ecommerce_AI import chatbot_response
from django.views import generic
from rest_framework import generics
from django.core.cache import cache

from django.shortcuts import render, redirect

from django.contrib.auth.hashers import check_password
from django.views import View
from django.views.decorators.csrf import ensure_csrf_cookie

import json

from django.http import JsonResponse


from .models import *
from .serializers import *

from rest_framework import viewsets, permissions
from rest_framework.decorators import action
from rest_framework.response import Response

class ChatbotResponse(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    
    @action(detail=False, methods=['post'])
    def post(self, request):
        data = request.data
        print("The user input is: ", data["user_input"])
        user_input = data['user_input']
        if user_input:
            response = chatbot_response(user_input)
            print("The response is: ", response)
            return Response({"response": response}, status=200)
        
        return Response({'error': 'No input provided'}, status=400)


class MenuOptionsView(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    
    # MenuOptionExtended.createMenusOptions()
    # MenuOptionExtended.delete_and_recreate_menuOption()
    
    @action(detail=False, methods=['post'])
    def post(self, request):
        
        data = request.data
        print(data["selected_option_id"])
        selected_option_id = data["selected_option_id"]
        
        if not selected_option_id:
            return Response({'error': 'No option ID provided'}, status=400) 
        
        # Fetch all menu options from the database
        
        try:
            # Get the sequence from the POST request
            selected_option_id = MenuOptionExtended.objects.get(menu_extended_id=selected_option_id)
            # Get the related options for the selected menu option
            next_options = selected_option_id.next_options.all()
            
            print("next option exist: ", next_options.exists())
            
            if next_options.exists():
            
                print(f"Next options for ID {selected_option_id.menu_extended_id}:  {next_options}")
                response_data = [
                {'menu_extended_id': option.menu_extended_id, 'menu_option_description': option.menu_option_description} for option in next_options]
                print("Response data:", response_data)
                return Response(response_data)
            else:
                print("No more options available")
                response_data = 'No more options available'
                return Response(response_data)
                # return Response({'data': 'No more options available'}, status=200)

            
        # if next_options:
            
        except MenuOptionExtended.DoesNotExist:
            return JsonResponse({'error': 'The selected option is not selected'}, status=400)

    
    @action(detail=False, methods=['get'])    
    def get(self, request):         
        all_options = list(MenuOptionExtended.get_all_menuOptions())
        total_option = len(all_options)  
        all_options_only_4 = all_options[:4]
        print("All options : ", all_options)
        return Response([
            {'menu_extended_id': option.menu_extended_id, 'menu_option_description': option.menu_option_description} for option in all_options_only_4
            ])


def is_ajax(request):
    return request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'

class MenuOptionEdit(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    
    def get(self, request):
        # Fetch all menu options from the database
        all_options = list(MenuOptionExtended.get_all_menuOptions())
        
        # Get 4 random options
        menu_options = all_options
        
        context = {
            'menu_options': menu_options,
        }
        
        query = request.GET.get('search', '') # Get the search term from the request
        
        if is_ajax(request) and query:
            menuOptions = MenuOptionExtended.objects.filter(productName__icontains=query) # Query the database
            
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

# PRODUCTS VIEWS

def is_ajax(request):
    return request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'

class ProductRegistration(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    
    @action(detail=False, methods=['get'])
    def get(self, request):
        print(request.data)
        query = request.data['search']
        # query = request.GE.get('search', '') # Get the search term from the request
        
        if is_ajax(request) and query:
            products = Product.objects.filter(productName__icontains=query) # Query the database
            
            product_names = list(products.values_list('name', flat=True).distinct()) # Get a list of product names
            print(f"Product names: {product_names}")
            return Response(product_names)
        
        return render(request, 'register/product.html')
    
    @action(detail=False, methods=['post'])
    def post(self, request):
        
        try:
            data = request.data

            product_name = data['productName']
            product_brand = data['productBrand']
            product_quantity_format = data['productQuantityFormat']
            product_section = data['productSection']
            price = data['productPrice']
            metric_unity = data['productMetricUnity']
            available = data['productAvailability']

            products = Product.get_all_products()
            
            product_name = product_name.strip()
            product_section = "Seção " + str(product_section)
            
            print("V1_PN: ", product_name, product_section, price, metric_unity, available)
            
            # THE RESULTS WERE REALLY BAD TO THE FIX OF PRODUCTS
            if not Product.objects.filter(name=product_name).exists():
                print("ENTER HERE")
                
                embeddings = Product.generate_emebed(product_name)

                product = Product(name=product_name, secao_produto=product_section, preco_venda=price, unidade_medida=metric_unity, disponivel=available, embeddings=embeddings, brand=product_brand, quantity=product_quantity_format)
                
                Product.fetch_products_alike(product_name)
                
                
                Product.generate_synonymous(product_name)
                product.save()
                
            else:
                print("ENTER THERE")
                
                products = Product.objects.filter(name=product_name)
                
                if len(products) > 1:
                    for product in products:
                        print("Product: ", product)
                        product_synonymous = Product.get_synonymus(product)
                        if not product_synonymous:
                            Product.generate_synonymous(product_list=product)
                        
                        
                        product.brand = product_brand
                        
                        print("Product original quantity: ", product.quantity)
                        print("Product quantity: ", product_quantity_format)
                    
                        product.quantity = product_quantity_format
                        product.save()
                else:
                    product_synonymous = Product.get_synonymus(product_name)
                    print("SYNONYMOUS: ", product_synonymous)
                    
                    product = Product.get_product_by_name(product_name)
                    
                    if not product_synonymous:
                        synonymous_list = Product.generate_synonymous(product_list=product)
                    else:
                        synonymous_list = Product.get_synonymus(product_name)
                        
                    print("Product name: ", product.name)
                    print("Product brand: ", product.brand)
                    print("Product quantity: ", product.quantity)
                    print("Product synonymous: ", product.synonymous)
            
                    product.brand = product_brand
                    
                    product.quantity = product_quantity_format
                    product.synonymous = synonymous_list
                    print("Product synonymous: ", product.synonymous)
                    
                    product.save()
                    
                    
        except Exception as err:
            print('Error here 2.0: ', err)
            return Response({'error': 'An error occured while registering the product: '}, status=500)
        
        products = Product.get_all_products()
        
        return Response([
            {'product_id': product.product_id, 'product_name': product.name} for product in products
            ])

class ProductListView(generic.ListView):
    model = Product
    context_object_name = "product_list"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        product_list = context['product_list']
        
        # Limiting the embeddings to first 3 elements
        for product in product_list:
            product.embeddings = product.embeddings[:3]
            
        return context
    
    template_name = "product_generic_list.html"

class ProductListCreateView(generics.ListCreateAPIView):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

class ProductRetrieveUpdateDestroyView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

