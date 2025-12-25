from collections import defaultdict
import json
def compute_numeric_averages(data):
    sums = defaultdict(float)
    counts = defaultdict(int)

    for item in data:
        for key, value in item.items():
            if isinstance(value, (int, float)):
                sums[key] += value
                counts[key] += 1

    return {key: sums[key] / counts[key] for key in sums}

with open("synt_bench\\result_metrics_e52.json", encoding="utf8") as data:
    data = json.load(data)
print(compute_numeric_averages(data))