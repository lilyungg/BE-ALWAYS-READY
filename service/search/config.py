import os

EMB_MODEL = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-large")
DATA_DIR = os.getenv("DATA_DIR", "e5-large")
INDEX_PATH = os.getenv("INDEX_PATH", "e5-large/merged.index")
META_PATH = os.getenv("META_PATH", "e5-large/merged_meta.pkl")
