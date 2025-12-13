import json
from prompt import PROMPT_GEN_DATASET
from tqdm import tqdm
import requests

INPUT_PATH = r"examples\merged.json"
OUTPUT_PATH = r"examples\rag_eval_dataset.json"

DEEPSEEK_API_KEY = ""


def generate_question(text: str) -> tuple[str, str]:
    try:
        messages = [
            {"role": "system", "content": PROMPT_GEN_DATASET},
            {"role": "user", "content": f"""ТЕКСТ:
\"\"\"
{text}
\"\"\""""},
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
        llm_res = json.loads(r.json()["choices"][0]["message"]["content"].strip('`').strip('json\n'))
        return llm_res["question"], llm_res["answer"]
    except Exception as e:
        return "", ""


dataset = []

with open(INPUT_PATH, encoding="utf-8") as f:
    documents = json.load(f)
idx = 0
for doc in tqdm(documents):
    question, answer = generate_question(doc["text"])
    if question == "":
        continue
    item = {
        "id": f"q_{idx:04d}",
        "question": question,
        "answers": [answer],
        "ground_truth_docs": [
            {
                "url": doc["url"],
                "text": doc["text"]
            }
        ]
    }

    dataset.append(item)
    idx += 1
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"Saved {len(dataset)} samples to {OUTPUT_PATH}")
