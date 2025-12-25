import asyncio
import re
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from config import *
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain_openai import ChatOpenAI

if MODEL_SOURCE == "deepseek":
    print("deepseek")
    def _deepseek_chat(messages: List[Dict[str, str]], temperature: float) -> str:
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")

        r = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": messages,
                "max_tokens": DEEPSEEK_MAX_TOKENS,
                "temperature": float(temperature),
            },
            timeout=DEEPSEEK_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


    def _invoke(prompt: ChatPromptTemplate, vars: Dict[str, Any], temperature: float) -> str:
        """Invoke DeepSeek with LangChain prompt messages."""
        lc_msgs = prompt.format_messages(**vars)
        ds_msgs: List[Dict[str, str]] = []

        for m in lc_msgs:
            if isinstance(m, SystemMessage):
                ds_msgs.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage):
                ds_msgs.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                ds_msgs.append({"role": "assistant", "content": m.content})
            else:
                ds_msgs.append({"role": "user", "content": str(getattr(m, "content", m))})

        return _deepseek_chat(ds_msgs, temperature=temperature)
elif MODEL_SOURCE == "openai":

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    _BASE_LLM = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        timeout=OPENAI_TIMEOUT,
        max_retries=2,
    )


    def _invoke(prompt: ChatPromptTemplate, vars: Dict[str, Any], temperature: float) -> str:
        lc_msgs = prompt.format_messages(**vars)
        llm = _BASE_LLM
        if OPENAI_MAX_TOKENS:
            llm = llm.bind(max_tokens=int(OPENAI_MAX_TOKENS))
        res = llm.invoke(lc_msgs)
        return (getattr(res, "content", str(res)) or "").strip()
else:
    raise ValueError("No correct MODEL_SOURCE")


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _dedup(xs: List[str]) -> List[str]:
    out, seen = [], set()
    for x in xs:
        x = (x or "").strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _hits_to_context(hits: List[dict]) -> str:
    parts = []
    for i, h in enumerate(hits or [], start=1):
        chunk = (h.get("chunk") or "").strip()
        if not chunk:
            continue
        title = h.get("document_title") or ""
        sec = h.get("section_title") or ""
        url = h.get("url") or ""
        header = f"[{i}] {title} | {sec}".strip()
        if url:
            header += f" | {url}"
        parts.append(header + "\n" + chunk)

    ctx = "\n\n---\n\n".join(parts).strip()
    if len(ctx) > MAX_CONTEXT_CHARS:
        ctx = ctx[: MAX_CONTEXT_CHARS - 3] + "..."
    return ctx


def _fallback_route(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ("собесед", "интерв", "interview", "mock", "тест", "вопросы")):
        return "interview"
    if any(k in t for k in ("курс", "план", "roadmap", "подготов", "учить", "программа")):
        return "course"
    return "qa"


ROUTER = ChatPromptTemplate.from_messages([
    ("system",
     "Ты роутер. Выбери сценарий: qa | interview | course.\n"
     "Верни СТРОГО JSON:\n"
     '{{"route":"qa|interview|course","topic":"","n_questions":10,"weeks":2,"level":"junior|middle|senior"}}\n'
     "Если тема не нужна — topic пустой. Никакого текста кроме JSON."
     ),
    ("human", "{text}")
])


def _route(text: str) -> Dict[str, Any]:
    raw = _invoke(ROUTER, {"text": text}, temperature=ROUTER_TEMP)
    data = _extract_json(raw) or {}
    route = (data.get("route") or "").strip().lower()
    if route not in ("qa", "interview", "course"):
        route = _fallback_route(text)
    return {
        "route": route,
        "topic": (data.get("topic") or "").strip(),
        "n_questions": int(data.get("n_questions") or 10),
        "weeks": int(data.get("weeks") or 2),
        "level": (data.get("level") or "middle").strip().lower(),
        "router_raw": raw,
    }


QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Ты помощник. Отвечай по-русски, кратко и структурно.\n"
     + ("ВАЖНО: отвечай ТОЛЬКО по контексту. Если ответа нет — так и скажи.\n" if RAG_ONLY else
        "Опирайся на контекст. Если его мало — можно добавить общие знания, но пометь их как 'вне контекста'.\n")
     ),
    ("human", "Вопрос:\n{question}\n\nКонтекст:\n{context}\n\nОтвет:")
])

INTERVIEW_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Ты интервьюер. Сделай тест и затем ответы.\n"
     "Формат:\n"
     "## Вопросы\n1)...\n\n"
     "## Ответы и разбор\n1)...\n\n"
     "В разборе ссылайся на фрагменты контекста [1],[2]...\n"
     + ("ВАЖНО: используй ТОЛЬКО контекст. Если чего-то нет — пиши 'нет в базе'.\n" if RAG_ONLY else "")
     ),
    ("human", "Тема: {topic}\nСделай {n} вопросов.\n\nКонтекст:\n{context}\n")
])

COURSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Ты наставник. Собери план подготовки к собеседованию.\n"
     "Нужно: цель, план по неделям (темы/практика/чек-лист), мини-проект в конце.\n"
     + ("ВАЖНО: используй ТОЛЬКО контекст и ссылки из него.\n" if RAG_ONLY else
        "Опирайся на контекст. Если ссылок нет — честно пометь это.\n")
     ),
    ("human", "Тема: {topic}\nУровень: {level}\nДлительность: {weeks} недель\n\nКонтекст:\n{context}\n")
])


def _rag_context(route: str, user_text: str, topic: str, search_client) -> Tuple[str, List[dict]]:
    n_queries = 3
    queries = _rewrite_queries(user_text, n=n_queries)

    # 2. Поиск по каждому запросу
    all_hits: List[dict] = []
    for qu in queries:
        try:
            hits = asyncio.run(search_client.search(qu, k=TOP_K))
            all_hits.extend(hits or [])
        except Exception:
            continue

    # 3. Дедуп по chunk + url
    seen = set()
    uniq_hits = []
    for h in all_hits:
        key = (
            (h.get("chunk") or "").strip(),
            (h.get("url") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq_hits.append(h)

    context = _hits_to_context(uniq_hits)

    return context, uniq_hits


VALIDATOR = ChatPromptTemplate.from_messages([
    ("system",
     "Ты валидатор. У тебя есть запрос, контекст из базы и черновик.\n"
     "Оставляй только то, что подтверждается контекстом. Если данных нет — так и скажи.\n"
     "Верни только финальный текст ответа."
     + ("" if RAG_ONLY else "\nЕсли добавляешь общее знание — явно пометь как 'вне контекста'.")
     ),
    ("human", "Запрос:\n{user_text}\n\nКонтекст:\n{context}\n\nЧерновик:\n{draft}\n\nФинальный ответ:")
])

FINALIZER = ChatPromptTemplate.from_messages([
    ("system", "Сделай ответ максимально читабельным: заголовки, списки, краткость. Не добавляй новые факты."),
    ("human", "{text}")
])


def _rewrite_queries(text: str, n: int = 5) -> List[str]:
    QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "Ты помощник для поиска по базе знаний.\n"
         "Сгенерируй несколько поисковых запросов.\n"
         "Раздели его на подзапросы, которые определяют части запроса"
         "Верни СТРОГО JSON без текста:\n"
         '{{"queries": ["...","..."]}}\n'
         ),
        ("human", "{text}")
    ])
    raw = _invoke(
        QUERY_REWRITE_PROMPT,
        {"text": text},
        temperature=0.3,
    )
    data = _extract_json(raw) or {}
    queries = data.get("queries") or []

    if not isinstance(queries, list):
        return [text]

    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    queries = queries[:n]

    # всегда добавляем исходный запрос
    return _dedup([text] + queries)


# MAIN
def agent_main(user_input: Union[str, Dict[str, Any]],
               search_client,
               history: Optional[Dict[str, Any]] = None) -> str:
    if isinstance(user_input, dict):
        user_text = str(user_input.get("text") or user_input.get("query") or user_input.get("message") or "").strip()
    else:
        user_text = str(user_input or "").strip()

    history_str = ""
    if history:
        try:
            history_str = json.dumps(history, ensure_ascii=False, indent=2)
        except Exception:
            history_str = str(history)
    orig_query = user_text
    if history_str:
        user_text = f"История переписки (json), можешь использовать для улучшения ответа на текущий запрос:\n{history_str}\n\nТекущий запрос:\n{user_text}".strip()

    if not user_text:
        return "Пустой запрос. Напиши вопрос или тему."

    r = _route(user_text)
    route, topic = r["route"], r["topic"]
    context, hits = _rag_context(route, user_text=orig_query, topic=topic, search_client=search_client)

    if RAG_ONLY and not context.strip():
        return "Не нашёл релевантной информации в базе по этому запросу. Попробуй переформулировать."
    if route == "interview":
        draft = _invoke(
            INTERVIEW_PROMPT,
            {"topic": topic or user_text, "n": r["n_questions"], "context": context or "(пусто)"},
            temperature=WORK_TEMP,
        )
    elif route == "course":
        draft = _invoke(
            COURSE_PROMPT,
            {"topic": topic or user_text, "weeks": r["weeks"], "level": r["level"], "context": context or "(пусто)"},
            temperature=WORK_TEMP,
        )
    else:
        draft = _invoke(
            QA_PROMPT,
            {"question": user_text, "context": context or "(пусто)"},
            temperature=WORK_TEMP,
        )

    checked = _invoke(
        VALIDATOR,
        {"user_text": user_text, "context": context or "(пусто)", "draft": draft},
        temperature=VALIDATOR_TEMP,
    )

    final_text = _invoke(FINALIZER, {"text": checked}, temperature=FINAL_TEMP).strip()

    if INCLUDE_SOURCES:
        if hits:
            final_text += "\n\nИсточники (из метаданных):\n" + "\n".join(
                f"{s['document_title']}- {s['url']}" for s in hits)
        elif RAG_ONLY:
            final_text += "\n\nИсточники: не найдены в метаданных."

    return final_text
