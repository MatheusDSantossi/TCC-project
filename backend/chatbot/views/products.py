from rest_framework import generics, permissions

from chatbot.models.product import Product
from chatbot.serializers import ProductSerializer
from users.models import CustomUser
from core.models import Business

from chatbot.services.product_ai import generate_synonyms_for, generate_embedding

from core.mixins import BusinessQuerysetMixin

class ProductListCreateView(BusinessQuerysetMixin, generics.ListCreateAPIView):
    permission_classes = [permissions.IsAuthenticated]
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
        
        embeddings = generate_embedding(name, business_id)
        
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
            
class ProductRetrieveUpdateDestroyView(BusinessQuerysetMixin, generics.RetrieveUpdateDestroyAPIView):
    # Scoped by BusinessQuerysetMixin: another business's product returns 404, not 403,
    # so the API does not reveal which product ids exist
    permission_classes = [permissions.IsAuthenticated]
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

