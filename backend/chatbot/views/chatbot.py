from rest_framework import status, viewsets
from chatbot.serializers import ChatbotRequestSerializer
from core.models import Business
from rest_framework.decorators import action
from rest_framework.response import Response
from chatbot.services.chatbot import generate_chatbot_response

from users.models import CustomUser
import io
import csv
import json
from typing import List
from django.db import transaction
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions, parsers
from chatbot.models import Product, Business
from django.contrib.postgres.fields import ArrayField
import pandas as pd
import regex as re


class ChatbotResponse(viewsets.GenericViewSet):
    """
    POST /api/chatbot/
    {
        "user_input": "...",
        "text_related_to": "...",   # optional
        "business_slug": "supermakert-santos"   # passed using either the url or localstorage information gotten from the user login
    }
    """

    serializer_class = ChatbotRequestSerializer
    permission_classes = []  # or [AllowAny]

    @action(detail=False, methods=["post"])
    def post(self, request):
        data = request.data

        # Validate input
        serializer = self.get_serializer(data=data)
        serializer.is_valid(raise_exception=True)

        print("CHATBOR RESPONSE: ", data)

        # Resolve business (or 404)
        business = serializer.validated_data.get("business_slug")
        print("Business in view: ", business)
        biz = Business.objects.get_id_by_slug(business)
        print("Business id in view: ", biz)

        # Delegate to chatbot service
        resp_text = generate_chatbot_response(
            user_iput=serializer.validated_data["user_input"],
            context=serializer.validated_data["text_related_to"],
            business_id=(biz if biz else None),
        )

        # Return a consistent, serializable paylaod
        return Response({"response": resp_text}, status=status.HTTP_200_OK)


class ProductImportView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]

    def post(self, request, *args, **kwargs):
        """
        Accepts a multipart file upload (.csv or .xlsx).
        Expected columns: name, brand, quantity, section, price, unit, stock, synonymous.
        synonymous can be a JSON array or pipe-separated string.
        """
        uploaded = request.FILES.get("file")

        if not uploaded:
            return Response({"detail", "No file"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user_email = self.request.user
            business = CustomUser.objects.get_business_for_email(user_email)

            print("User email: ", user_email)
            print("business: ", business)

            if business:
                print(f"Associated business: {business.name}")
            else:
                print("No valid invite found for this email")
        except Exception as e:
            business = None
            print(f"Error retrieving business: {str(e)}")

        # Accept business id in form
        business_id = Business.objects.get_id_by_name(business)
        business = business_id

        # if business_id:
        #     try:
        #         business = Business.objects.get(pk=int(business_id))

        #     except:
        #         business = None

        # Read into pandas for convenience (handles csv/xlsx)
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded, dtype=str).fillna("")
            else:
                df = pd.read_excel(uploaded, dtype=str).fillna("")
        except Exception as e:
            return Response({"detail": f"Failed to read file: {str(e)}"}, status=400)

        products_to_create = []
        errors = []

        # define mapping / normalize column names
        def getcell(row, keys):
            for k in keys:
                if k in row and row[k] != "":
                    return row[k]
            return ""

        for idx, row in df.to_dict(orient="records"):
            pass

        # simpler loop
        for i, raw in enumerate(df.to_dict(orient="records")):
            try:
                name = getcell(raw, ["name", "nome", "productName"]).strip()
                brand = getcell(raw, ["brand", "marca"]).strip()
                quantity = getcell(raw, ["quantity", "quantidade"]).strip()
                section_raw = getcell(raw, ["section", "secao", "seção"])
                section = int(section_raw) if section_raw else 0
                price_raw = getcell(raw, ["price", "preco", "preço"])
                price = float(price_raw.replace(",", ".").strip()) if price_raw else 0.0
                unit = getcell(raw, ["unit", "unidade"]).strip()
                stock_raw = getcell(raw, ["stock", "estoque"])
                stock = float(stock_raw) if stock_raw else 0.0
                syn_cell = getcell(raw, ["synonymous", "sinonimos", "synonyms"]) or ""
                # parse synonyms: try JSON then fallback to separators
                synonyms = []
                if syn_cell:
                    try:
                        parsed = json.loads(syn_cell)
                        if isinstance(parsed, list):
                            synonyms = [str(x) for x in parsed]

                    except Exception:
                        # splitting by pipes or semicolons or commas (carefully)
                        synonyms = [
                            s.strip() for s in re.split(r"[|;,]", syn_cell) if s.strip()
                        ]
                if not name:
                    errors.append({"row": i + 1, "error": "Missing name"})
                    continue

                product = Product(
                    business=business,
                    name=name,
                    brand=brand,
                    quantity=quantity,
                    secao_produto=section,
                    preco_venda=price,
                    unidade_medida=unit,
                    disponivel=stock,
                    synonymous=synonyms,  # if field is jsonb or ArrayField adapt accordingly
                )
                products_to_create.append(product)

            except Exception as e:
                errors.append({"row": i + 1, "error": str(e)})
        # Bulk create in chunks
        created = 0

        try:
            with transaction.atomic():
                BATCH = 500
                for i in range(0, len(products_to_create), BATCH):
                    batch = products_to_create[i : i + BATCH]
                    Product.objects.bulk_create(batch, batch_size=BATCH)
                    created += len(batch)
        except Exception as e:
            return Response({"detail": "DB error: " + str(e)}, status=500)

        return Response({"created": created, "errors": errors})
