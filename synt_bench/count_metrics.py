import requests
from tqdm import tqdm

from rag_deepseek import MergedFaissIndex, DeepSeekRag
import json

from synt_bench.metrics import evaluate_sample

INDEX_PATH = "data\merged.index"
META_PATH = "data\merged_meta.pkl"
DEEPSEEK_API_KEY = ""
ix = MergedFaissIndex()

ix.load()
rag = DeepSeekRag(ix)

with open("examples\\rag_eval_dataset.json", encoding="utf8") as data:
    data = json.load(data)


def llm_call(msg):
    messages = [
        {"role": "system", "content": "Answer using only the provided context."},
        {"role": "user", "content": f"{msg}"},
    ]
    r = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-chat",
            "messages": messages,
            "max_tokens": 256,
            "temperature": 0.2,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


meow = []
for i in tqdm(data):
    ans, hits = rag.answer(i, k=3)
    meow.append(evaluate_sample(i, hits, ans, llm_call, 3))
    with open("result_metrics_e52.json", "w", encoding="utf-8") as f:
        json.dump(meow, f, ensure_ascii=False, indent=4)
