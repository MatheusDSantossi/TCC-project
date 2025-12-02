from rest_framework import generics

from chatbot.models.product import Product
from chatbot.serializers import ProductSerializer
from users.models import CustomUser
from core.models import Business

from chatbot.services.product_ai import generate_synonyms_for, generate_embedding

from core.mixins import BusinessQuerysetMixin

# TODO: I'm getting this error when trying to create a product:
# IntegrityError at /api/products/
# null value in column "product_id" of relation "products_with_embeddings" violates not-null constraint
# DETAIL:  Failing row contains (null, BISCOITO RECHEADO  COCO, Seção3, 1.99, PT, TRELOSO, 30X120G, 3, 100.00, [-0.02286109, 0.022939662, 0.018616376, 0.011887718, -0.00251269..., ["BISCOITINHO", "COQUINHO"]).

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
            
class ProductRetrieveUpdateDestroyView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

