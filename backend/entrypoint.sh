#!/usr/bin/env bash
set -euo pipefail

# Print date for logs
echo ">>> entrypoint: starting at $(date -u)"

# Wait a little
# sleep 1

# Run migrations and collectstatic (fail loudly so logs show errors)
echo ">>> Running migrations"
python manage.py migrate --noinput

echo ">>> Collecting static files"
python manage.py collectstatic --noinput --clear

# We can do other startup tasks here (create superuser if env flags present etc.)

# Exec gunicorn replacing shell
PORT="${PORT:-8000}"
echo ">>> Starting gunicorn on 0.0.0.0:${PORT}"
exec gunicorn --bind 0.0.0.0:"${PORT}" chatbot_project.wsgi
