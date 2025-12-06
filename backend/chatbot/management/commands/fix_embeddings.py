import re
import ast
import logging
from typing import List, Optional, Tuple

from django.core.management.base import BaseCommand
from django.db import transaction

from chatbot.models import Product
from core.models import Business

from utils.embeddings import embed_fn
from utils.project_ecommerce_AI import genai, find_section

logger = logging.getLogger(__name__)

# regex to capture floats (including exponent)
FLOAT_RE = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?")

def parse_embedding_value(val) -> List[float]:
    """
    Parse a stored embedding value into a list of floats.
    Accepts:
      - Python list/tuple of numbers
      - Postgres array literal string like '{0.1,0.2,-0.3}'
      - String with python list: '[0.1, -0.2, ...]'
      - String that contains numbers separated by commas or spaces
      - Possibly double-quoted string '"{0.1,0.2}"' (strip quotes)
    Returns list of floats or raises ValueError if it cannot parse.
    """
    if val is None:
        return []
    
    # if already list/tuple of numbers
    if isinstance(val, (list, tuple)):
        return [float(x) for x in val]
    
    # if bytes, decode
    if isinstance(val, bytes):
        val = val.decode("utf-8", errors="ignore")
        
    if not isinstance(val, str):
        # try simple coercion
        try:
            return [float(val)]
        except Exception:
            raise ValueError(f"Unrecognized embeddings type: {type(val)}")
        
    s = val.strip()
    
    # strip outer quotes if present
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
        
    # cases:
    # 1) Postgres array literal: {0.1, 0.2, -0.3}
    if s.startswith("{") and s.endswith("}"):
        inner = s[1:-1].strip()
        
        # split by comma but allow spaces
        if inner == "":
            return []
        
        parts = [x.strip() for x in inner.split(",")]
        try:
            return [float(x) for x in parts if x != ""]
        except Exception as e:
            # fall through to numeric regex fallback
            pass

    # 2) Python list literal: [0.1, -0.2,...]
    if s.startswith("[") and s.endswith("]"):
        try:
            lst = ast.literal_eval(s)
            if isinstance(lst, (list, tuple)):
                return [float(x) for x in lst]
        except Exception:
            # fall through to numeric regex fallback
            pass
    
    # 3) fallback: extract numbers with regex
    nums = FLOAT_RE.findall(s)
    if nums:
        return [float(x) for x in nums]
    
    # nothing parsed
    raise ValueError(f"Could not parse embeddings from string:  {repr(val)[:200]}")

def fix_length(vec: List[float], target_dim: int, mode: str = "error") -> Tuple[List[float], bool]:
    """
    Ensure vec has target_dim length.
    mode: "error" -> raise ValueError if mismatch
          "pad"   -> pad with zeros (if short) or trim (if long)
          "trim"  -> trim if long, error if short
    Returns (new_vec, changed_flag)
    """
    if target_dim is None:
        return vec, False
    
    n = len(vec)
    
    if n == target_dim:
        return vec, False
    
    if mode == "error":
        raise ValueError(f"Embedding length {n} does not match target {target_dim}")
    
    if mode == "pad":
        if n < target_dim:
            return vec + [0.0] * (target_dim - n), True
        
        else:
            return vec[:target_dim], True
        
    if mode == "trim":
        if n > target_dim:
            return vec[:target_dim], True
        
        else:
            raise ValueError(f"Embedding length {n} less than target {target_dim}")
    
    raise ValueError("Unknown mode for fix_length")

def call_embed_fn_safe(name_text: str, biz_obj: Optional[Business] = None, sleep_seconds: float = 0.0):
    """
    Attempt to call embed_fn with a couple different signatures to be robust.
    Returns a list[float] or raises.
    """
    if embed_fn is None:
        raise RuntimeError("embed_fn is not defined")
    
    # small pause to prevent rate limits
    # if sleep_seconds and sleep_seconds > 0:
    #     time.sleep(sleep_seconds)

    if biz_obj is None:
        raise
    
    emb = embed_fn(find_section, genai, biz_obj, name_text)
    
    # Convert numpy arrays/other to plain list of floats
    try:
        # if it's numpy array or list-like
        return [float(x) for x in emb]
    except Exception as e:
        raise RuntimeError(f"embed_fn returned unexpected result: {type(emb)} -> {e}")

        

class Command(BaseCommand):
    help = "Normalize embeddings column (parse inconsistent formats) and optionally pad/trim lengths."
    
    def add_arguments(self, parser):
        parser.add_argument("--business-id", type=int, required=False, help="Filter to this business id")
        parser.add_argument("--dry-run", action="store_true", help="Do not write changes, only report")
        parser.add_argument("--expected-dim", type=int, required=False, help="Expected embedding dimension (e.g., 1536). If set, mismatches are handled according to --fix-mode")
        parser.add_argument("--fix-mode", choices=["error", "pad", "trim"], default="error", help="What to do if embedding dimension mismatches (default: error)")
        parser.add_argument("--limit", type=int, default=None, help="Process only first N rows (for testing)")
        
        parser.add_argument("--recompute", action="store_true", help="Recompute embeddings using embed_fn and overwrite existing embeddings")
        parser.add_argument("--biz-slug", type=str, required=False, help="Business slug to pass to embed_fn (if needed)")
        parser.add_argument("--batch-size", type=int, default=200, help="How many records to bulk_update at once")
        parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between embedding calls (throttle)")
        
    def handle(self, *args, **options):
        business_id = options.get("business_id")
        dry_run = options.get("dry_run")
        expected_dim = options.get("expected_dim")
        fix_mode = options.get("fix_mode")
        limit = options.get("limit")
        
        recompute = options.get("recompute", False)
        biz_slug = options.get("biz_slug")
        batch_size = options.get("batch_size", 200)
        sleep_seconds = options.get("sleep", 0.0)
        
        qs = Product.objects.all().order_by("product_id")
        
        if business_id:
            qs = qs.filter(business_id=business_id)
            
        total = qs.count()
        
        if limit:
            qs = qs[:limit]
            
         # get business instance if slug provided (for embed_fn signature that needs it)
        biz_obj = None
        if biz_slug:
            try:
                biz_obj = Business.objects.get(slug=biz_slug)
            except Business.DoesNotExist:
                self.stdout.write(self.style.ERROR(f"Business with slug '{biz_slug}' not found."))
                return
            
        self.stdout.write(f"Processing {qs.count()} products (total in filter: {total})")
        
        parsed_count = 0
        updated_count = 0
        skipped = 0
        errors = 0
        mismatches = 0
        
        to_update_batch = []
        
        for product in qs.iterator():
            parsed_count += 1
            raw = product.embeddings
            
            try:
                if recompute:
                # recompute from scratch using embed_fn
                    try:
                        new_emb = call_embed_fn_safe(product.name, biz_obj=biz_obj, sleep_seconds=sleep_seconds)
                    except Exception as e:
                        self.stdout.write(self.style.ERROR(f"[{product.product_id}] embed recompute failed: {e}"))
                        logger.exception("embed recompute failed for product %s", product.product_id)
                        errors += 1
                        continue
                    
                # decide whether to save
                need_write = True
                # compare to existing quickly
                try:
                    old_parsed = parse_embedding_value(product.embeddings)
                    if old_parsed == new_emb:
                        need_write = False
                except Exception:
                    # old couldn't parse -> we will write
                    need_write = True
                    
                if need_write:
                        self.stdout.write(f"[{product.product_id}] will {'(dry) ' if dry_run else ''}update embedding (len {len(new_emb)}) via recompute")
                        product.embeddings = new_emb
                        to_update_batch.append(product)
                        if len(to_update_batch) >= batch_size:
                            if not dry_run:
                                with transaction.atomic():
                                    Product.objects.bulk_update(to_update_batch, ["embeddings"])
                                updated_count += len(to_update_batch)
                            else:
                                updated_count += len(to_update_batch)
                            to_update_batch = []
                        else:
                            skipped += 1

                else:
                    # just normalize parsing/formats (no recompute)
                    try:
                        vec = parse_embedding_value(product.embeddings)
                    except Exception as e:
                        self.stdout.write(self.style.ERROR(f"[{product.product_id}] parse error: {e}"))
                        errors += 1
                        continue

                    # length enforcement
                    if expected_dim is not None and len(vec) != expected_dim:
                        n = len(vec)
                        if fix_mode == "pad" and n < expected_dim:
                            vec = vec + [0.0] * (expected_dim - n)
                        elif fix_mode == "trim" and n > expected_dim:
                            vec = vec[:expected_dim]
                        else:
                            self.stdout.write(self.style.ERROR(f"[{product.product_id}] length mismatch {n} != {expected_dim}, mode={fix_mode}; skipping"))
                            mismatches += 1
                            continue

                    # decide whether to save normalized list
                    need_write = not isinstance(product.embeddings, list) or len(product.embeddings) != len(vec) or any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(vec, product.embeddings if isinstance(product.embeddings, list) else []))
                    if need_write:
                        self.stdout.write(f"[{product.product_id}] will {'(dry) ' if dry_run else ''}update embeddings (len {len(vec)}) by normalization")
                        product.embeddings = vec
                        to_update_batch.append(product)
                    else:
                        skipped += 1

                    if len(to_update_batch) >= batch_size:
                        if not dry_run:
                            with transaction.atomic():
                                Product.objects.bulk_update(to_update_batch, ["embeddings"])
                            updated_count += len(to_update_batch)
                        else:
                            updated_count += len(to_update_batch)
                        to_update_batch = []

            except Exception as e:
                errors += 1
                logger.exception("Processing failed for product %s", getattr(product, "product_id", "<unknown>"))
                self.stdout.write(self.style.ERROR(f"[{product.product_id}] unexpected error: {e}"))
                continue

        # flush remaining batch
        if to_update_batch:
            if not dry_run:
                with transaction.atomic():
                    Product.objects.bulk_update(to_update_batch, ["embeddings"])
                updated_count += len(to_update_batch)
            else:
                updated_count += len(to_update_batch)

        self.stdout.write(self.style.SUCCESS(
            f"Done. processed: {parsed_count}, updated (or would update): {updated_count}, skipped: {skipped}, parse_errors: {errors}, mismatches: {mismatches}"
        ))
                    
               
