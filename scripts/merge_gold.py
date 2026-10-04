"""Merge synthetic candidate cards and adjudicated labels; split groups in time order."""
import argparse
import collections
from common import index_rows, read_jsonl, write_jsonl


def merge(rows, labels):
    gold = index_rows(labels)
    if set(index_rows(rows)) != set(gold):
        raise ValueError("candidate and gold ids must match exactly")
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["state_card"].splitlines()[0]].append(row)
    order = sorted(groups, key=lambda g: (min(row["ts"] for row in groups[g]), g))
    count = 0; result = []; n = len(rows)
    for group in order:
        split = "dev" if count < int(n * 0.70) else "sel" if count < int(n * 0.85) else "test"
        for row in groups[group]:
            result.append({**row, **gold[row["id"]], "split": split})
        count += len(groups[group])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", required=True); p.add_argument("--gold", required=True); p.add_argument("--out", required=True)
    a = p.parse_args(); rows = merge(read_jsonl(a.candidates), read_jsonl(a.gold)); write_jsonl(a.out, rows)
    print("merged", len(rows), "rows", dict(collections.Counter(r["split"] for r in rows)))


if __name__ == "__main__":
    main()
