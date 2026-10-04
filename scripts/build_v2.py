"""Public rename augmentation: both classes draw from the same configurable name pool."""
import argparse
import copy
import random
import re
from pathlib import Path
from battery import TEMPLATE
from common import boolean, read_jsonl, write_jsonl


def rename(state, new_name, header=r"^(goal:)[^\n]*"):
    changed, count = re.subn(header, lambda match: match.group(1) + " " + new_name, state, count=1)
    if count != 1:
        raise ValueError("state must begin with the configured name header")
    return changed


def augment(rows, pool, seed, negative_copies, template=TEMPLATE, header=r"^(goal:)[^\n]*"):
    if not pool:
        raise ValueError("empty name pool")
    rng = random.Random(seed); result = copy.deepcopy(rows)
    for row in rows:
        if boolean(row["questions"][template]["label"]):
            for _ in range(2):
                clone = copy.deepcopy(row); clone["state"] = rename(row["state"], rng.choice(pool), header); result.append(clone)
    negatives = [r for r in rows if not boolean(r["questions"][template]["label"])]
    for row in rng.sample(negatives, min(negative_copies, len(negatives))):
        clone = copy.deepcopy(row); clone["state"] = rename(row["state"], rng.choice(pool), header)
        clone["questions"][template]["label"] = False; result.append(clone)
    rng.shuffle(result)
    return result


def build(source, out, pool, seed=0, prefix="changelog-merge", header=r"^(goal:)[^\n]*", template=TEMPLATE):
    source, out = Path(source), Path(out)
    for filename, count in (("bat_train.jsonl", 70), ("bat_heldout.jsonl", 14)):
        rows = read_jsonl(source / filename)
        write_jsonl(out / filename, augment(rows, pool, seed, count, template, header))
    rows = read_jsonl(source / "labels_battery.jsonl")
    test = [{"id": r["id"], "state": r["state"], "tpl": boolean(r["gold"][template]), "judgment": boolean(r["gold"]["judgment"])} for r in rows if r["split"] == "test"]
    ordinary = [name for name in pool if prefix not in name]
    decoys = [name for name in pool if prefix in name]
    if not ordinary or not decoys:
        raise ValueError("evaluation pool needs both template-style and ordinary names")
    rng = random.Random(seed + 1)
    swap = [{**r, "state": rename(r["state"], rng.choice(ordinary if r["tpl"] else decoys), header)} for r in test]
    write_jsonl(out / "eval_test.jsonl", test); write_jsonl(out / "eval_swap.jsonl", swap)
    print("wrote synthetic augmented training and paired evaluation files")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--names", required=True); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--template-prefix", default="changelog-merge"); p.add_argument("--template-question", default=TEMPLATE)
    p.add_argument("--header", default=r"^(goal:)[^\n]*")
    a = p.parse_args(); pool = [line.strip() for line in Path(a.names).read_text().splitlines() if line.strip()]
    build(a.data, a.out, pool, a.seed, a.template_prefix, a.header, a.template_question)


if __name__ == "__main__":
    main()
