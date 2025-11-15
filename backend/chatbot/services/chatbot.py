from .product_ai import chatbot_response
# from celery import shared_task
# from celery_progress.backend import ProgressRecorder  # optional helper


def generate_chatbot_response(user_iput: str, context: str, business_id: int):
    # any pre- or post-processing can live here
    return chatbot_response(user_iput, context, business_id)


# @shared_task(bind=True)
# def import_products_task(self, filepath, user_id, business_id):
#     progress = ProgressRecorder(self)  # records progress in backend
#     # total = estimate_total_steps(filepath)  # implement this function based on your logic
#     processed = 0

#     # open file, iterate rows
#     for row in iterate_rows(filepath):
#         processed += 1
#         if total:
#             progress.set_progress(
#                 processed, total, description=f"Processed {processed}/{total}"
#             )
#         else:
#             # set arbitrary progress, e.g. percentage by file pointer, or every N rows
#             progress.set_progress(processed, 200, description=f"Processed {processed}")
#     return {"created": ..., "errors": ...}
