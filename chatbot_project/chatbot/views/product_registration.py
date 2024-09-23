from django.shortcuts import render, redirect

from django.contrib.auth.hashers import check_password
from django.views import View

from ..models.product import *
from django.http import JsonResponse

def is_ajax(request):
    return request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'

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
        productName = request.POST.get('productNameInput')
        product_section = request.POST.get('productSectionInput')
        price = request.POST.get('productPriceInput')
        metric_unity = request.POST.get('productMetricUnityInput')
        available = request.POST.get('productAvailableInput')
        
        products = Product.get_all_products()
        
        # if Product.objects.filter(nome=productName).exists():
        #     # If product is already on the database
        #     return render(request, 'chatbot/index.html', {
        #         'error_message': f"Product {productName} already exists!"
        #     })
        
        productName = productName.strip()
        
        print("V1_PN: ", productName, product_section, price, metric_unity, available)
        
        # productName = Product.fix_product_name(productName)
        
        # print("V2_PN: ", productName, product_section, price, metric_unity, available)
        # THE RESULTS WERE REALLY BAD TO THE FIX OF PRODUCTS
        if not Product.objects.filter(nome=productName).exists():
            
            embeddings = Product.generate_emebed(productName)
            
            # print(embeddings)
            
            product = Product(nome=productName, secao_produto=product_section, preco_venda=price, unidade_medida=metric_unity, disponivel=available, embeddings=embeddings)
            
            # Product.objects.create(product.nome,)
            
            Product.fetch_products_alike(productName)
            
            product.save()
            
            # print("ALL SIMILAR PRODUCTS: ", Product.generate_synonymous(productName))
            Product.generate_synonymous(productName)
            
            # for product in products:
            #     print(product.nome, product.preco_venda, product.secao_produto)
        else:
            product_synonymous = Product.get_synonymus(productName)
            print("SYNONYMOUS: ", product_synonymous)
            if not product_synonymous:
                Product.generate_synonymous(productName)
            else:
                print("PRODUCT ALREADY HAVE SYNONYMOUS: ", product_synonymous)
            
        return redirect('index')
