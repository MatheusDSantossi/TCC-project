from django.apps import AppConfig
from django.db.utils import OperationalError

class ChatbotConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "chatbot"
    
    def ready(self):
        try:
            from .models import MenuOptionExtended
            
            if MenuOptionExtended.objects.exists():
                return
            # Only create if table is empty
            if MenuOptionExtended.objects.count() == 0:
                MenuOptionExtended.createMenusOptions()
        except OperationalError:
            # DB não está pronto ainda (startup/migrate) — pular
            return
        except Exception:
            # evitar que qualquer outra exceção quebre o processo de inicialização
            return