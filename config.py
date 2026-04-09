import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "instaclaim-dev-key-2024")
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))  # 10MB
