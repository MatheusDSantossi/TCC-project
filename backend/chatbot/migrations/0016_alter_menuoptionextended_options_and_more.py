from django.db import migrations, models

def copy_menu_description(apps, schema_editor):
    MenuOptionExtended = apps.get_model("chatbot", "MenuOptionExtended")
    # copy old -> new column
    for obj in MenuOptionExtended.objects.all():
        obj.description = obj.menu_option_description
        obj.save(update_fields=["description"])

class Migration(migrations.Migration):
    dependencies = [
        ("chatbot", "0015_convert_embeddings_to_json"),
    ]
    operations = [
        # 1) add the new field as nullable & non-unique
        migrations.AddField(
            model_name="menuoptionextended",
            name="description",
            field=models.CharField(max_length=255, null=True),
        ),
        # 2) copy data over
        migrations.RunPython(copy_menu_description, reverse_code=migrations.RunPython.noop),
        
        # 3) make it non-nullable & unique
        migrations.AlterField(
            model_name="menuoptionextended",
            name="description",
            field=models.CharField(max_length=255, unique=True, null=False),
        ),
        # 4) remove the old column
        migrations.RemoveField(
            model_name="menuoptionextended",
            name="menu_option_description",
        ),
        # 5) set your ordering option
        migrations.AlterModelOptions(
            name="menuoptionextended",
            options={"ordering": ["description"]},
        ),
    ]
