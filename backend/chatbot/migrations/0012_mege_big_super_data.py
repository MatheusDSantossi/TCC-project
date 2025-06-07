# backend/chatbot/migrations/0011_merge_big_super_data.py

from django.db import migrations

def merge_big_super_products(apps, schema_editor):
    """
    This function will run as part of the migration. It copies each row from
    products_with_embeddings_bigger into products_with_embeddings, field by field.
    """
    # Get the historical (state‐aware) versions of the two models:
    Product         = apps.get_model('chatbot', 'Product')
    BigSuperProduct = apps.get_model('chatbot', 'BigSuperProduct')

    # Loop over every BigSuperProduct row and create a new Product row
    for big in BigSuperProduct.objects.all():
        Product.objects.create(
            business       = big.business,
            name           = big.name,
            brand          = big.brand,
            quantity       = big.quantity,
            secao_produto  = big.secao_produto,
            preco_venda    = big.preco_venda,
            unidade_medida = big.unidade_medida,
            disponivel     = big.disponivel,
            embeddings     = big.embeddings,
            synonymous     = big.synonymous or [],
        )


def reverse_merge(apps, schema_editor):
    """
    If you ever roll back this migration, delete all Product rows that came from
    the BigSuper table. We identify them by the presence of a non‐null 'code' field.
    """
    Product = apps.get_model('chatbot', 'Product')
    # If 'code' was only present in BigSuperProduct, we use that to clean up.
    Product.objects.filter(code__isnull=False).delete()


class Migration(migrations.Migration):

    dependencies = [
        ("chatbot", "0011_convert_disponivel_to_decimal"),
    ]

    operations = [
        migrations.RunPython(merge_big_super_products, reverse_merge),
    ]
