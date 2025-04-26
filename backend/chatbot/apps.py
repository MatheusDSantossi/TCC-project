from django.apps import AppConfig


class ChatbotConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "chatbot"
    
    def ready(self):
        from .models import MenuOptionExtended
        # Only create if table is empty
        if MenuOptionExtended.objects.count() == 0:
            MenuOptionExtended.createMenusOptions()