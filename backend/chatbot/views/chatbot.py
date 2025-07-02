from rest_framework import status, viewsets

from chatbot.serializers import ChatbotRequestSerializer
from core.models import Business

from rest_framework.decorators import action
from rest_framework.response import Response

from chatbot.services.chatbot import generate_chatbot_response

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
