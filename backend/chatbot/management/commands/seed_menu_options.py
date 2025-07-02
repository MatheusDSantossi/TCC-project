from django.core.management.base import BaseCommand
from chatbot.managers.menuoption import MenuOptionManager
from chatbot.models import MenuOptionExtended

class Command(BaseCommand):
    help = "Seed or refresh all chatbot menu options"
    
    def handle(self, *args, **options):
        
        MenuOptionExtended.objects.update_or_create(
            description="Quais os produtos o Supermercado Santos vende?",
            defaults={"description": "Quais são os produtos que vocês vendem?"}
        )
        
        
        manager = MenuOptionManager()
        # MenuOptionExtended.objects.all().delete()
        
        # uma lista de perguntas (sugestões de perguntas com interrogação in the end?) uma lista com tudo que it's able to answer
        # Options definitions
        # level1 = [
        #     'Tenho dúvida de um produto específico',
        #     'Quero informações do estabelecimento',
        #     'Quero saber mais sobre os serviços do estabelecimento',
        #     'Quais são os produtos que vocês vendem?',
        #     'Desejo outra opção'
        # ]
        
        # # Create or reuse level 1
        # opts1 = [manager.get_or_create_description(d) for d in level1]
        
        # # Level 2 for each:
        # sub_opt = {
        #     opts1[0]: ['Digitar o nome do produto', 'Digitar a seção do produto', 'Digitar a marca do produto', 'Digitar alguma descrição do produto ou sinônimo'],
        #     opts1[1]: ['Nome do gerente', 'Digitar a pergunta sobre o mercado'],
        #     opts1[2]: ['Quais são as promoções do dia?', 'Quais são as promoções do dia?'],
        #     opts1[3]: ['Lista de categorias disponíveis', 'Quero saber sobre um produto específico'],
        #     opts1[4]: ['Informar o assunto sobre o qual quero tratar'],
        # }
        
        # # Clear & link relations:
        # for parent, children in sub_opt.items():
        #     parent.next_options.clear()
        #     parent.next_options.add(*[manager.get_or_create_description(d) for d in children])
            
        # self.stdout.write(self.style.SUCCESS("Menu options seeded."))
    