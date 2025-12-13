def mrr(retrieved_urls, relevant_urls):
    for i, url in enumerate(retrieved_urls, start=1):
        if url in relevant_urls:
            return 1 / i
    return 0.0


def recall_at_k(retrieved_urls, relevant_urls, k):
    retrieved_k = set(retrieved_urls[:k])
    relevant = set(relevant_urls)
    return len(retrieved_k & relevant) / len(relevant) if relevant else 0.0


def recall_at_k(retrieved_urls, relevant_urls, k):
    retrieved_k = set(retrieved_urls[:k])
    relevant = set(relevant_urls)
    return len(retrieved_k & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(retrieved_urls, relevant_urls, k):
    retrieved_k = retrieved_urls[:k]
    relevant = set(relevant_urls)
    return sum(1 for url in retrieved_k if url in relevant) / k


import math


def ndcg_at_k(retrieved_urls, relevant_urls, k):
    dcg = 0.0
    for i, url in enumerate(retrieved_urls[:k], start=1):
        if url in relevant_urls:
            dcg += 1 / math.log2(i + 1)

    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant_urls), k) + 1))
    return min(dcg / idcg if idcg > 0 else 0.0, 1)


PROMPT_LLM_JUDGE = """
Ты — строгий эксперт.

Оцени ответ модели по шкале от 1 до 5:

5 — полностью правильный и полный
4 — правильный, но есть неточности
3 — частично правильный
2 — в основном неправильный
1 — полностью неправильный

Вопрос:
{question}

Эталонный ответ:
{reference_answer}

Ответ модели:
{generated_answer}

Ответь ТОЛЬКО числом от 1 до 5.
"""


import re

def parse_judge_score(text: str) -> int:
    match = re.search(r"[1-5]", text)
    if not match:
        raise ValueError(f"Invalid judge output: {text}")
    return int(match.group())

def llm_as_judge(llm_call, question, ref_answer, gen_answer):
    prompt = PROMPT_LLM_JUDGE.format(
        question=question,
        reference_answer=ref_answer,
        generated_answer=gen_answer
    )

    raw = llm_call(prompt)  # <- твой API
    return parse_judge_score(raw)



def evaluate_sample(
    sample,
    retrieved_docs,
    generated_answer,
    llm_call,
    k=5
):
    retrieved_urls = [d["url"] for d in retrieved_docs]
    relevant_urls = [d["url"] for d in sample["ground_truth_docs"]]

    metrics = {
        "mrr": mrr(retrieved_urls, relevant_urls),
        "recall@k": recall_at_k(retrieved_urls, relevant_urls, k),
        "precision@k": precision_at_k(retrieved_urls, relevant_urls, k),
        "ndcg@k": ndcg_at_k(retrieved_urls, relevant_urls, k),
    }

    judge_score = llm_as_judge(
        llm_call,
        sample["question"],
        sample["answers"][0],
        generated_answer
    )

    metrics["llm_judge_score"] = judge_score

    return metrics

def average_metrics(results):
    avg = {}
    for key in results[0]:
        avg[key] = sum(r[key] for r in results) / len(results)
    return avg
