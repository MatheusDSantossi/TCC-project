from django.shortcuts import render, redirect

from django.contrib.auth.hashers import check_password
from django.views import View

from ..models.product import *
from django.http import JsonResponse

from django.views.decorators.csrf import csrf_exempt

import json

def is_ajax(request):
    return request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'

# DON'T USE THAT IN PROD
@csrf_exempt
class ProductRegistration(View):
    
    def get(self, request):
        query = request.GET.get('search', '') # Get the search term from the request
        
        if is_ajax(request) and query:
            products = Product.objects.filter(productName__icontains=query) # Query the database
            
            # product_names = list(products.values_list('productName', flat=True).distinct()) # Get a list of product names
            # product_names = list(products.values('productName', flat=True).distinct()) # Get a list of product names
            product_names = list(products.values('nome', flat=True).distinct()) # Get a list of product names
            return JsonResponse(product_names, safe=False)
        
        return render(request, 'register/product.html')
    
    
    
    def post(self, request):
        try:
            data = json.loads(request.body)

            productName = data.get('productName')
            product_section = data.get('productSection')
            price = data.get('productPrice')
            metric_unity = data.get('productMetricUnity')
            available = data.get('productAvailability')
            
            products = Product.get_all_products()
            
            productName = productName.strip()
            
            print("V1_PN: ", productName, product_section, price, metric_unity, available)
            
            # THE RESULTS WERE REALLY BAD TO THE FIX OF PRODUCTS
            if not Product.objects.filter(nome=productName).exists():
                
                embeddings = Product.generate_emebed(productName)

                product = Product(nome=productName, secao_produto=product_section, preco_venda=price, unidade_medida=metric_unity, disponivel=available, embeddings=embeddings)
                
                Product.fetch_products_alike(productName)
                
                product.save()
                
                Product.generate_synonymous(productName)
                
            else:
                product_synonymous = Product.get_synonymus(productName)
                print("SYNONYMOUS: ", product_synonymous)
                if not product_synonymous:
                    Product.generate_synonymous(productName)
                else:
                    print("PRODUCT ALREADY HAVE SYNONYMOUS: ", product_synonymous)
                    
        except Exception as err:
            print('Error: ', err)
            return JsonResponse({'error': 'An error occured while registering the product: '}, status=500)
            
        return redirect('index')
