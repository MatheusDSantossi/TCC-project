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
from chatbot.models import Product
from core.models import Business
from django.contrib.postgres.fields import ArrayField
import pandas as pd
import regex as re
import traceback


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

        print("=== ProductImportView called ===")
        print("Incoming content-type:", request.content_type)
        print("FILES keys:", list(request.FILES.keys()))
        print("META CONTENT_LENGTH:", request.META.get("CONTENT_LENGTH"))

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

        business_obj = None

        # If `business` is already a Business instance (best case), we use it
        if isinstance(business, Business):
            business_obj = business
        else:
            # if business is a name string, try to get by name
            try:
                if isinstance(business, str) and business:
                    business_obj = Business.objects.get(name=business)
                else:
                    # fallback: if id was earlier computed we use it
                    business_id = Business.objects.get_id_by_name(business) if isinstance(business, str) else None
                    
                    if business_id:
                        business_obj = Business.objects.get(pk=business_id)
            except Business.DoesNotExist:
                business_obj = None

        # Read into pandas for convenience (handles csv/xlsx)
        try:

            print("Uploaded file name:", uploaded.name)
            print("Uploaded file size (bytes):", getattr(uploaded, "size", "unknown"))

            if uploaded.name.endswith(".csv"):
                tried = []
                df = None
                # default read
                try:
                    df = pd.read_csv(uploaded, dtype=str).fillna("")
                    tried.append("default")
                except Exception as e1:
                    print("Default CSV read failed:", str(e1))
                    tried.append("default failed: " + str(e1))

                    # rewind file pointer for next attemp
                    try:
                        uploaded.seek(0)
                    except Exception:
                        pass

                # try latin-1 encoding (common with Excel/Windows exports)
                try:
                    df = pd.read_csv(uploaded, dtype=str, encoding="latin-1").fillna("")
                    tried.append("latin-1")
                except Exception as e2:
                    print("Latin-1 CSV read failed:", str(e2))
                    tried.append("latin-1 failed: " + str(e2))

                    try:
                        uploaded.seek(0)
                    except Exception:
                        pass

                    # try semicolon separator
                    try:
                        df = pd.read_csv(uploaded, dtype=str, sep="|").fillna("")
                        tried.append("sep='|'")

                    except Exception as e3:
                        print("Separator '|' CSV read failed:", str(e3))
                        tried.append("sep='|' failed: " + str(e3))
                if df is None:
                    raise ValueError(
                        "unable to parse CSV. Attempts: " + "; ".join(tried)
                    )
            else:
                try:
                    df = pd.read_excel(uploaded, dtype=str, engine="openpyxl").fillna(
                        ""
                    )
                except Exception as ex:
                    print("read_excel failed:", str(ex))
                    # try without engine fallback
                    df = pd.read_excel(uploaded, dtype=str).fillna("")

            print("DataFrame loaded. shape:", getattr(df, "shape", None))
            print("DataFrame head:\n", df.head(5).to_dict(orient="records"))

        except Exception as e:
            print("Exception while reading/parsing file:")
            return Response({"detail": f"Failed to read file: {str(e)}"}, status=400)

        products_to_create = []
        errors = []

        # define mapping / normalize column names
        def getcell(row, keys):
            for k in keys:
                if k in row and row[k] != "":
                    return row[k]
            return ""

        # for idx, row in df.to_dict(orient="records"):
        #     pass

        # simpler loop
        for i, raw in enumerate(df.to_dict(orient="records")):
            try:
                name = getcell(raw, ["name", "nome", "productName", "Produto"]).strip()
                brand = getcell(raw, ["brand", "marca", "Marca"]).strip()
                quantity = getcell(raw, ["quantity", "quantidade"]).strip()
                section = getcell(raw, ["section", "secao", "seção", "secao_produto"])
                price_raw = getcell(raw, ["price", "preco", "preço", "preco_venda", "Preço de venda"])
                price = float(price_raw.replace(",", ".").strip()) if price_raw else 0.0
                unit = getcell(raw, ["unit", "unidade", "unidade_medida", "Unidade"]).strip()
                stock_raw = getcell(raw, ["stock", "estoque", "disponivel", "Estoque"])
                stock = float(stock_raw) if stock_raw else 0.0
                syn_cell = getcell(raw, ["synonymous", "sinonimos", "synonyms"]) or ""
                embeddings = getcell(raw, ["embeddings", "embedding"]) or ""
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
                
                if not business_obj:
                    return Response({" detail": "Business not found for this user"}, status=status.HTTP_400_BAD_REQUEST)

                product = Product(
                    business=business_obj,
                    name=name,
                    brand=brand,
                    quantity=quantity,
                    section=section,
                    price=price,
                    unit=unit,
                    stock=stock,
                    synonymous=synonyms,  # if field is jsonb or ArrayField adapt accordingly
                    embeddings=embeddings,
                )
                products_to_create.append(product)

                print("product: ", product)

            except Exception as e:
                errors.append({"row": i + 1, "error": str(e)})
                print(f"Error processing row {i+1}: {e}")
                traceback.print_exc()

        print("Prepared products count:", len(products_to_create))
        print("Row-level errors:", errors[:10])
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
            print("Exception during bulk_create:")
            return Response({"detail": "DB error: " + str(e)}, status=500)

        print("Import finished. created:", created)
        return Response({"created": created, "errors": errors})
