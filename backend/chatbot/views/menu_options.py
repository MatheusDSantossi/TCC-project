from chatbot.models import MenuOptionExtended

from rest_framework import viewsets, permissions
from rest_framework.decorators import action
from rest_framework.response import Response

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

