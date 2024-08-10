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
        
        print("V1_PN: ", productName, product_section, price, metric_unity, available)
        
        # productName = Product.fix_product_name(productName)
        
        # print("V2_PN: ", productName, product_section, price, metric_unity, available)
        # THE RESULTS WERE REALLY BAD TO THE FIX OF PRODUCTS
        
        embeddings = Product.generate_emebed(productName)
        
        # print(embeddings)
        
        product = Product(nome=productName, secao_produto=product_section, preco_venda=price, unidade_medida=metric_unity, disponivel=available, embeddings=embeddings)
        
        # Product.objects.create(product.nome,)
        product.save()
        
        print(Product.generate_synonimus(productName))
        
        # for product in products:
        #     print(product.nome, product.preco_venda, product.secao_produto)
        
        return redirect('index')
