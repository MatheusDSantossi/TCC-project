# chatbot/migrations/0004_alter_product_disponivel.py
from django.db import migrations

class Migration(migrations.Migration):
    dependencies = [
        ("chatbot", "0003_alter_product_synonymous"),
    ]

    operations = [
       migrations.RunSQL(
            # forward: boolean → numeric(10,2)
            sql="""
                ALTER TABLE products_with_embeddings
                ALTER COLUMN disponivel
                  TYPE numeric(10,2)
                  USING (CASE WHEN disponivel THEN 1.00 ELSE 0.00 END);
            """,
            # reverse: numeric → boolean
            reverse_sql="""
                ALTER TABLE products_with_embeddings
                ALTER COLUMN disponivel
                  TYPE boolean
                  USING (disponivel <> 0);
            """,
        ),
    ]
