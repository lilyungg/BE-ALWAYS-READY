import os


def get_bool_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {
        "1", "true", "yes", "y", "on"
    }


def get_str_env(name: str, default: str = "") -> str | None:
    return os.getenv(name, default).strip()


# === Model source ===
MODEL_SOURCE = get_str_env("MODEL_SOURCE", "openai").lower()

if MODEL_SOURCE not in {"openai", "deepseek"}:
    raise ValueError(
        f"Invalid MODEL_SOURCE='{MODEL_SOURCE}'. "
        "Allowed values: 'openai', 'deepseek'"
    )


# === External services ===
SEARCH_SERVICE_URL = get_str_env(
    "SEARCH_SERVICE_URL",
    "http://agent:8000",
)


# === RAG settings ===
RAG_ONLY = get_bool_env("RAG_ONLY", "1")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
INCLUDE_SOURCES = get_bool_env("INCLUDE_SOURCES", "1")
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "2000"))


# === Temperatures ===
ROUTER_TEMP = float(os.getenv("ROUTER_TEMP", "0.0"))
WORK_TEMP = float(os.getenv("WORK_TEMP", "0.2"))
VALIDATOR_TEMP = float(os.getenv("VALIDATOR_TEMP", "0.0"))
FINAL_TEMP = float(os.getenv("FINAL_TEMP", "0.2"))


# === DeepSeek ===
DEEPSEEK_API_KEY = get_str_env("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = get_str_env("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = get_str_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_TIMEOUT = float(os.getenv("DEEPSEEK_TIMEOUT", "60"))
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "2000"))


# === OpenAI ===
OPENAI_API_KEY = get_str_env("OPENAI_API_KEY", "")
OPENAI_MODEL = get_str_env("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = get_str_env("OPENAI_BASE_URL")
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "60"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
