"""Deterministic dataset sampling policies for comparable experiments."""

from collections import defaultdict
import random


def select_records(records, count, strategy="stratified", seed=42):
    records = list(records or [])
    count = max(0, min(int(count), len(records)))
    if count >= len(records):
        return records
    if strategy == "prefix":
        return records[:count]
    rng = random.Random(seed)
    if strategy == "random":
        indices = list(range(len(records)))
        rng.shuffle(indices)
        return [records[index] for index in indices[:count]]
    if strategy != "stratified":
        raise ValueError(f"Unknown selection strategy: {strategy}")

    groups = defaultdict(list)
    for index, record in enumerate(records):
        key = (
            str(record.get("phenomenon", "unknown")),
            str(record.get("label", "unknown")),
        )
        groups[key].append((index, record))
    for rows in groups.values():
        rng.shuffle(rows)

    selected = []
    ordered_keys = sorted(groups)
    while len(selected) < count:
        progressed = False
        for key in ordered_keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return [record for _, record in selected]
