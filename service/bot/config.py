import os

BOT_TOKEN = os.getenv(
    "BOT_TOKEN",
    ""
)

POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    ""
)

AGENT_SERVICE_URL = os.getenv(
    "AGENT_SERVICE_URL",
    "http://agent:8010"
)
SEARCH_SERVICE_URL = os.getenv(
    "SEARCH_SERVICE_URL",
    "http://agent:8000"
)
