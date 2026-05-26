from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_ROOT = Path(__file__).resolve().parent

OPENAI_API_KEY = "EMPTY"
BASE_URL = "http://127.0.0.1:1234/v1"
EMBEDDING_MODEL_PATH = str(PROJECT_ROOT / "bge-large-zh-v1.5")
PRODUCT_VECTOR_PATH = str(AGENT_ROOT / "product_vector_store")
ORDER_DB_PATH = str(AGENT_ROOT / "ecommerce_orders.db")

# Flask config
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False
