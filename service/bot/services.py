from typing import Dict, Any

import httpx
from config import *


async def send_to_external_service(text: str, history: dict[str, str]):
    async with httpx.AsyncClient() as client:
        print(history)
        print(AGENT_SERVICE_URL)
        response = await client.post(
            f'{AGENT_SERVICE_URL}/agent',
            json={"text": text, "history": history},
            timeout=70
        )
        return response.json()["result"]


async def parse_url(url: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{SEARCH_SERVICE_URL}/parse",
            json={"url": url}
        )
        r.raise_for_status()
        return r.json()
