from pathlib import Path
import sys
import os
from dotenv import load_dotenv
from datetime import timedelta

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Ensure the root directory is included
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(encoding="UTF-8")

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
# No fallback on purpose: a missing key must stop startup, not run on a known default
SECRET_KEY = os.environ["DJANGO_SECRET_KEY"]

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get("DEBUG", "0") == "1"

def parse_allowed_hosts(env_value: str):
    # Accept comma-separated hosts. Keep only non-empty entries.
    if not env_value:
        return []
    return [h.strip() for h in env_value.split(",") if h.strip()]

ALLOWED_HOSTS = parse_allowed_hosts(os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1"))


# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.sites",
    "rest_framework",
    "users",
    "chatbot",
    "core",
    "knox",
    "corsheaders",
    "django_rest_passwordreset",
    "allauth",
    "allauth.account",
    "allauth.socialaccount",
    "allauth.socialaccount.providers.google",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "allauth.account.middleware.AccountMiddleware",
    
]

ROOT_URLCONF = "chatbot_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR/"templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "chatbot_project.wsgi.application"

REST_FRAMEWORK = {
    # "DEFAULT_AUTHENTICATION_CLASSES": ('rest_framework_simplejwt_authentication.JWTAuthentication',),
    'DEFAULT_AUTHENTICATION_CLASSES': ('knox.auth.TokenAuthentication',),
    # Fail closed: a view that forgets permission_classes requires a login instead
    # of being public. Public views must declare AllowAny explicitly.
    'DEFAULT_PERMISSION_CLASSES': ('rest_framework.permissions.IsAuthenticated',),
}

# SIMPLE_JWT = {
#     'ACCESS_TOKEN_LIFETIME': timedelta(minutes=5),
#     'REFRESH_TOKEN_LIFETIME': timedelta(days=1),
#     # 'ALGORITHM': 'HS256',
# }

# Database
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql_psycopg2",
        "NAME": os.environ.get("DATABASE_DB"),
        "USER": os.environ.get("DATABASE_USER"),
        "PASSWORD": os.environ.get("DATABASE_PASSWORD"),
        "HOST": os.environ.get("DATABASE_HOST"),
        "PORT": os.environ.get("DATABASE_PORT"),
        # Persistent connections: reuse one connection per worker for up to 60s
        # instead of opening a new one per request; the health check reconnects
        # after Postgres restarts or drops an idle connection
        "CONN_MAX_AGE": 60,
        "CONN_HEALTH_CHECKS": True,
    }
}

# Password validation
# https://docs.djangoproject.com/en/5.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.1/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.1/howto/static-files/

STATIC_URL = "/static/"
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Default primary key field type
# https://docs.djangoproject.com/en/5.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

CORS_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost",
]

# IMPORTANT: This is for testing purposes only. Do not use in production.
CORS_ORIGIN_ALLOW_ALL = True

# CORS_ALLOWS_CREDENTIONS = True

# AUTHENTICATION_BACKENDS = (
#     "knox.auth.TokenAuthentication",
#     "django.contrib.auth.backends.ModelBackend",
#     "allauth.account.auth_backends.AuthenticationBackend",
# )

# CSRF_TRUSTED_ORIGINS = [
#     "http://localhost:5173",
#     "http://localhost:5174"
# ]

AUTH_USER_MODEL = "users.CustomUser"

AUTHENTICATION_BACKENDS = [
    # 'users.auth_backend.EmailAuthBackend'
    "django.contrib.auth.backends.ModelBackend",
    # `allauth` specific authentication methods, such as login by email
    'allauth.account.auth_backends.AuthenticationBackend',
]

# Email auth backend configuration
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_HOST_USER = os.environ.get("USER_EMAIL")
EMAIL_HOST_PASSWORD = os.environ.get("EMAIL_HOST_PASSWORD")
EMAIL_USE_TLS = True
EMAIL_USE_SSL = False
DEFAULT_FROM_EMAIL = 'CHATBOT HECATE'

# Setting the SITE_ID (usually 1)
SITE_ID = 1

# Login and logout
# LOGIN_REDIRECT_URL = '/callback/'

# Google Stuff
SOCIALACCOUNT_PROVIDERS = {
    "google": {
        "SCOPE": ['profile', 'email'],
        "AUTH_PARAMS": {
            "access_type": "online",
        },
        "APP": {
          "client_id": os.environ.get("GOOGLE_OAUTH_CLIENT_ID"),
          "secret": os.environ.get("GOOGLE_OAUTH_SECRET_KEY"), 
        },
        "OAUTH_PKCE_ENABLED": True,
        "FETCH_USERINFO": True,
    }
}

# SOCIALACCOUNT_STORE_TOKEN = True 

GOOGLE_OAUTH_CLIENT_ID = os.environ.get('GOOGLE_OAUTH_CLIENT_ID')
if not GOOGLE_OAUTH_CLIENT_ID:
    raise ValueError("GOOGLE_OAUTH_CLIENT_ID not set in environment variables")

# These lines below are needed to allow the Google sing in popup to show up
SECURE_REFERRER_POLICY = 'no-referrer-when-downgrade'
SECURE_CROSS_ORIGIN_OPENER_POLICY = 'same-origin-allow-popups'
