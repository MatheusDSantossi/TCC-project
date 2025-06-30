from django.shortcuts import render
from django.http import JsonResponse
from utils.project_ecommerce_AI import chatbot_response
from rest_framework import generics, status
from django.core.cache import cache

from django.shortcuts import render, redirect

import json

from django.http import JsonResponse

from .models import BusinessDetail, MenuOptionExtended, Product
from.serializers import BusinessDetailSerializer, ProductSerializer, ChatbotRequestSerializer
from users.models import CustomUser
from core.models import Business

from rest_framework import viewsets, permissions
from rest_framework.decorators import action
from rest_framework.response import Response

from core.mixins import BusinessQuerysetMixin

from .services.product_ai import generate_synonyms_for, generate_embedding

from .services.business import resolve_business
from.services.chatbot import generate_chatbot_response

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
class ChatbotResponse(viewsets.GenericViewSet):
    """
    POST /api/chatbot/
    {
        "user_input": "...",
        "text_related_to": "...",   # optional
        "business_slug": "supermakert-santos"   # passed using either the url or localstorage information gotten from the user login
    }
    """
    serializer_class = ChatbotRequestSerializer
    permission_classes = [] # or [AllowAny]
    
    @action(detail=False, methods=['post'])
    def post(self, request):
        data = request.data
        
        # Validate input
        serializer = self.get_serializer(data=data)
        serializer.is_valid(raise_exception=True)
        
        print("CHATBOR RESPONSE: ", data)
        
        # Resolve business (or 404)
        business = serializer.validated_data.get("business_slug")
        print("Business in view: ", business)
        biz = Business.objects.get_id_by_slug(business)
        print("Business id in view: ", biz)
        
        # Delegate to chatbot service
        resp_text = generate_chatbot_response(
            user_iput=serializer.validated_data["user_input"],
            context=serializer.validated_data["text_related_to"],
            business_id=(biz if biz else None)
        )
        
        # Return a consistent, serializable paylaod
        return Response({"response": resp_text}, status=status.HTTP_200_OK)

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
          
        sample = all_options[:5]
        print("All options : ", all_options)
        return Response([
            {'menu_extended_id': option.menu_extended_id, 'description': option.description} for option in sample
            ])

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

