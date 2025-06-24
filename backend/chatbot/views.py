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
from users.models import CustomUser
from core.models import Business
from .serializers import *

from rest_framework import viewsets, permissions
from rest_framework.decorators import action
from rest_framework.response import Response

from core.mixins import BusinessQuerysetMixin
from core.permissions import IsBusinessMember

from .services.product_ai import generate_synonyms_for, generate_embedding

def is_ajax(request):
    return request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'

# Business
class BusinessDetailListCreateView(BusinessQuerysetMixin, generics.ListCreateAPIView):
    """
    GET /api/business-detail/ -> list details for request.user.business
    POST /api/business-detail/ -> create a new detail under request.user.business
    """
    # permission_classes = [permissions.IsAuthenticated]
    permission_classes = [permissions.IsAuthenticated]
    queryset = BusinessDetail.objects.all()
    serializer_class = BusinessDetailSerializer
    
    def perform_create(self, serializer):
        # Automatically set business = request.user.business
        serializer.save(business=self.request.user.business)
        
class BusinessDetailRetrieveUpdateDestroyView(BusinessQuerysetMixin, generics.RetrieveUpdateDestroyAPIView):
    """
    GET    /api/business-detail/<pk>/    → retrieve one detail (must belong to request.user.business)
    PUT    /api/business-detail/<pk>/    → update it
    DELETE /api/business-detail/<pk>/    → delete it
    """
    permission_classes = [permissions.IsAuthenticated]
    queryset = BusinessDetail.objects.all()
    serializer_class = BusinessDetailSerializer


# Chatbot views
class ChatbotResponse(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    
    @action(detail=False, methods=['post'])
    def post(self, request):
        data = request.data
        
        print("CHATBOR RESPONSE: ", data)
        
        user_input = data.get('user_input')
        text_related_to = data.get('text_related_to')
        business = data.get('business_slug')
        
        if not user_input:
            return Response({'error': 'No input provided'}, status=400)
        
        # Validating business
        if business:
            # I'm checking if the business was sent, because I use chatbot
            # in two differents places and in one of them the slug isn't send.
            # TODO: It's interesting to change this approach
            try:
                 biz = Business.objects.get(slug=business)
            except Business.DoesNotExist:
                 return Response({'error': f'Unkown business: {business}'}, status=404)
        
    
        # Call the chatbot_response logic
        response = chatbot_response(user_input, text_related_to, biz.id)
                    
        print("The response is: ", response)
        
        return Response({"response": response}, status=200)


class MenuOptionsView(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]

    @action(detail=False, methods=['post'])
    def post(self, request):
        """If no `selected_option_id` provided, return only the *root* menu options. Otherwise return the children of the selected option."""
        
        sel_id = request.data.get("selected_option_id")
        
        data = request.data
        print(data["selected_option_id"])
        selected_option_id = data["selected_option_id"]
        
        if sel_id is None:
            # Root nodes = those with no parents
            roots = MenuOptionExtended.objects.filter(previous_options__isnull=True).order_by("menu_extended_id")
            
            payload = [
                {"menu_extended_id": opt.menu_extended_id, "menu_extended_id": opt.description} for opt in roots
            ]
            
            return Response(payload)
        
        # ---- Subsequent load: look up children
        
        try:
            parent = MenuOptionExtended.objects.get(pk=sel_id)
            
            
        except MenuOptionExtended.DoesNotExist:
            return Response({'error': 'The selected option ID is invalid'}, status=400)

        children = parent.next_options.all().order_by("menu_extended_id")
        
        if not children:
            return Response({"message": "No more options available"})
        
        payload = [
            {"menu_extended_id": opt.menu_extended_id, "description": opt.description} for opt in children
        ]
        
        return Response(payload)
    
    @action(detail=False, methods=['get'])    
    def get(self, request):         
        all_options = list(MenuOptionExtended.objects.all().order_by("menu_extended_id"))
        # total_option = len(all_options)
          
        sample = all_options[:5]
        print("All options : ", all_options)
        return Response([
            {'menu_extended_id': option.menu_extended_id, 'description': option.description} for option in sample
            ])

class MenuOptionEdit(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    
    def get(self, request):
        # Fetch all menu options from the database
        all_options = list(MenuOptionExtended.objects.all().order_by("menu_extended_id"))
        
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
# class ProductRegistration(BusinessQuerysetMixin, viewsets.ViewSet):
#     # permission_classes = [permissions.AllowAny]
#     permission_classes = [IsBusinessMember, permissions.AllowAny]
    
#     queryset = Product.objects.all()
#     serializer_class = ProductSerializer
    
#     @action(detail=False, methods=['get'])
#     def get(self, request):
#         print(request.data)
#         query = request.data['search']
#         # query = request.GE.get('search', '') # Get the search term from the request
        
#         if is_ajax(request) and query:
#             products = Product.objects.filter(productName__icontains=query) # Query the database
            
#             product_names = list(products.values_list('name', flat=True).distinct()) # Get a list of product names
#             print(f"Product names: {product_names}")
#             return Response(product_names)
        
#         return render(request, 'register/product.html')
    
#     @action(detail=False, methods=['post'])
#     def post(self, request):
        
#         try:
#             data = request.data

#             product_name = data['productName']
#             product_brand = data['productBrand']
#             product_quantity_format = data['productQuantityFormat']
#             product_section = data['productSection']
#             price = data['productPrice']
#             metric_unity = data['productMetricUnity']
#             available = data['productAvailability']
#             synonyms = data['productSynonyms'] # Array with some synonyms the employee choose

#             print("Synonyms: ", synonyms)

#             # products = Product.get_all_products()
            
            
#             product_name = product_name.strip()
#             product_section = "Seção " + str(product_section)
            
#             print("V1_PN: ", product_name, product_section, price, metric_unity, available)
            
#             existing_product = Product.objects.filter(name=product_name, brand=product_brand).first()
            
#             # if not Product.objects.filter(name=product_name).exists() or not Product.objects.filter(brand=product_brand).exists():
#             if not existing_product:
#                 print("Creating new Product...")
                
#                 embeddings = Product.generate_emebed(product_name)

#                 product = Product(name=product_name, secao_produto=product_section, preco_venda=price, unidade_medida=metric_unity, disponivel=available, embeddings=embeddings, brand=product_brand, quantity=product_quantity_format)
                
#                 Product.fetch_products_alike(product_name)
                
                
#                 synonymous = Product.generate_synonymous(product=product_name, brand=product_brand, synonyms=synonyms)
                
#                 # product.brand = product_brand
#                 product.synonymous = synonymous
#                 print("Product original quantity: ", product.quantity)
#                 print("Product quantity: ", product_quantity_format)
#                 print("Product brand: ", product.brand)
            
#                 # product.quantity = product_quantity_format
#                 product.save()
                
#                 result_product = product
                
#             else:
#                 print("Updating existing product...")
                
#                 product = existing_product
                
#                 product.brand = product_brand
#                 product.quantity = product_quantity_format
                
#                 product_synonymous = Product.get_synonymus(product_name)
                
#                 if not product_synonymous:
#                     synonymous = Product.generate_synonymous(product=product_name, brand=product_brand, synonyms=synonyms)
                    
#                 else:
#                     product.synonymous = product_synonymous
                    
#                 product.save()
#                 result_product = product
                    
#         except Exception as err:
#             print('Error here 2.0: ', err)
#             return Response({'error': 'An error occured while registering the product: '}, status=500)
        
#         # products = Product.get_all_products()
        
#         return Response([
#             {'product_id': result_product.product_id, 'product_name': result_product.name}
#             ])
#         # return Response([
#         #     {'product_id': product.product_id, 'product_name': product.name} for product in products
#         #     ])

# class ProductUpdate(viewsets.ViewSet):
#     @action(detail=True, methods=['put'], url_path='update-product/(?P<pk>[^/.]+)')
#     def update_product(self, request, pk):
#         try:
#             product = Product.objects.get(product_id=pk)
#             data = request.data
#             print("Data product: ", data)
#             product.name = data.get('productName', product.name)
#             product.brand = data.get('productBrand', product.brand)
#             product.quantity = data.get('productQuantityFormat', product.quantity)
#             product.secao_produto = data.get('productSection', product.secao_produto)
#             product.preco_venda = data.get('productPrice', product.secao_produto)
#             product.unidade_medida = data.get('productMetricUnity', product.unidade_medida)
#             product.disponivel = data.get('productAvailability', product.disponivel)
#             product.synonymous = data.get('productLocalSynonyms', product.synonymous)
#             print("Product brand: ", product.synonymous)
#             print("Product synonyms: ", data.get('productLocalSynonyms'))
            
#             product.save()
            
#             return Response({'message': 'Product updated successfully'}, status=201)
        
#         except Product.DoesNotExist:
#             return Response({'error': 'Product not found'}, status=404)

# class ProductDelete(viewsets.ViewSet):
#     @action(detail=True, methods=['delete'], url_path='delete-product/(?P<pk>[^/.]+)')
#     def delete_product(self, request, pk):
#         try:
#             product = Product.objects.get(product_id=pk)
#             product.delete()
#             return Response({'message': 'Product deleted successfully'}, status=204)
        
#         except Product.DoesNotExist:
#             return Response({'error': 'Product not found'}, status=404)

# class ProductListView(generic.ListView):
#     model = Product
#     context_object_name = "product_list"

#     def get_context_data(self, **kwargs):
#         context = super().get_context_data(**kwargs)
#         product_list = context['product_list']
        
#         # Limiting the embeddings to first 3 elements
#         for product in product_list:
#             product.embeddings = product.embeddings[:3]
            
#         return context
    
#     template_name = "product_generic_list.html"

# TODO: How can I modify this, because for example, I want to add synonyms when it doesn't have. In this case, I'll need to call my generate_synonmys function and edit the product before saving it.
class ProductListCreateView(BusinessQuerysetMixin, generics.ListCreateAPIView):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    
    def perform_create(self, serializer):
        
        try:
            user_email = self.request.user
            business = CustomUser.objects.get_business_for_email(user_email)
            
            print("User email: ", user_email)
            print("business: ", business)
            
            if business:
                print(f"Associated business: {business.name}")
            else:
                print("No valid invite found for this email")
        except Exception as e:
            business = None
            print(f"Error retrieving business: {str(e)}")
        
        business_id = Business.objects.get_id_by_name(business)
        
        # Grabbing the raw name/brand (and any user-passed synonyms)
        name = serializer.validated_data["name"]        
        brand = serializer.validated_data.get("brand", "")       
        section = serializer.validated_data["section"]
        new_section = "Seção" + str(section)
        incoming = serializer.validated_data.get("synonymous", [])        
        
        embeddings = generate_embedding(name)
        
        instance = serializer.save(business=business, embeddings=embeddings, section=new_section)
        
        # Generate the final synonyms
        generated = generate_synonyms_for(product=name, brand = brand, synonyms=incoming, business=business_id)
        
        # Checking if the current synonymous exists
        if incoming:
            # Saving with the business and the computed list
            instance.synonymous = generated
        
        else:
            # Saving with the business and the same synonymous
            instance.synonymous = incoming
            
class ProductRetrieveUpdateDestroyView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

