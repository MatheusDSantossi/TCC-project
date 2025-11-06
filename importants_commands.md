# Important commmands

## Django

OBS: This command is to fill the table with the products based on the xlsx file and during this process it creates the embeddings.
python manage.py import_big_super path/to/file.xlsx --business big-super

- This one is responsible for creating a new business

python manage.py shell

from django.contrib.auth import get_user_model
User = get_user_model()
u = User.objects.get(username="yourusername")
Business.objects.create(name="My Store", slug="my-store", owner=u)

## Docker

docker-compose down -v --remove-orphans
docker system prune -a --volumes

## Database (SQL)

If some problem occurs I have this backup "backup_2025-06-10.dump".
I can restore using:
pg_restore --clean --no-owner -d project_pcc_v3 backup_2025-06-10.dump

## Python Tutorial about login with google

Here's the link of the [tutorial](https://youtu.be/38CflJ837xI?t=1162).
