#!/usr/bin/env python3
"""Zero-training decision readout harness. Asks an OpenAI-compatible chat
endpoint one constrained question per item, reads top_logprobs of the FIRST
generated token, keeps only allowed option tokens, renormalizes via softmax,
and scores calibration. No generation past 2 tokens. Stdlib only."""
import argparse, hashlib, json, math, os, sys, time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Set B labels (APPLY/REJECT) are compared against options (Yes/No); map the
# label to the option space first. Set A labels are already single letters and
# pass through LABEL_MAP.get unchanged.
LABEL_MAP = {"APPLY": "Yes", "REJECT": "No"}
SYS_A = "You are a strict router. You answer with exactly one capital letter."
SYS_B = ("Apply a reviewer finding if it identifies a factual error or an unsupported "
         "claim within the draft's scope. Reject it if its premise is wrong, "
         "it is outside scope, or it is already handled.")

# ---- HTTP ------------------------------------------------------------------

def http_post(base, payload, timeout=120):
    """Single POST attempt; raises on any error. Retry lives in
    fetch_distribution so semantic failures (empty logprobs / bad distribution)
    are retried alongside transport errors."""
    url = base.rstrip("/") + "/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

# ---- Logprob matching / distribution --------------------------------------

def build_distribution(raw_tlp, options):
    """For each option keep MAX logprob over tokens equal after strip+lower;
    unmatched = min(observed) - 3.0. Softmax over allowed options only."""
    if not raw_tlp:
        return None, None
    logs = [t.get("logprob", 0.0) for t in raw_tlp]
    base_min = min(logs)
    best = {}
    for opt in options:
        key = opt.strip().lower()
        cand = [t.get("logprob") for t in raw_tlp
                if t.get("token", "").strip().lower() == key]
        best[opt] = max(cand) if cand else (base_min - 3.0)
    m = max(best.values())
    exps = {o: math.exp(best[o] - m) for o in options}
    s = sum(exps.values())
    if s <= 0 or not math.isfinite(s):
        return None, None
    return {o: exps[o] / s for o in options}, best

def score_dist(dist):
    items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    top, top_p = items[0]
    second = items[1][1] if len(items) > 1 else 0.0
    ent = -sum(p * math.log(p) for p in dist.values() if p > 0)
    return top, top_p, top_p - second, ent

# ---- Question builders -----------------------------------------------------

def question_A(item, reverse=False):
    """Choice: label candidates A,B,C..., final escalate option. Returns
    (sys, user, options, gt_letter, name_by_letter). Truncates to the first 25
    candidates so the escalate letter is always available."""
    cands = list(item.get("candidates", []))
    if len(cands) > 25:
        if not reverse:
            print(f"[warn] item {item.get('id')}: {len(cands)} candidates, "
                  f"truncating to 25", file=sys.stderr)
        cands = cands[:25]
    if reverse:
        cands = list(reversed(cands))
    n = len(cands)
    letters = [LETTERS[i] for i in range(n)]
    lines, name_by_letter = [], {}
    for i, c in enumerate(cands):
        L = letters[i]; name_by_letter[L] = c["name"]
        lines.append(f"{L}. {c['name']} — {c.get('title','')}: {c.get('description','')}")
    esc = LETTERS[n]
    lines.append(f"{esc}. None of these fits; escalate to a human.")
    user = (item["prompt"].strip() + "\n\n" + "\n".join(lines) +
            "\n\nWhich ONE option should handle this task? Answer with the single letter only.")
    expect = item.get("expect", "__none__")
    gt = next((letters[i] for i, c in enumerate(cands) if c["name"] == expect), esc)
    return SYS_A, user, letters + [esc], gt, name_by_letter

def question_B(item):
    user = (f"Reviewer finding: {item['finding']}\n\n"
        "Should the author APPLY this finding? Answer Yes or No.")
    return SYS_B, user, ["Yes", "No"]

# ---- Per-item processing ---------------------------------------------------

def request_payload(model, sysmsg, user):
    return {"model": model,
               "messages": [{"role": "system", "content": sysmsg},
                            {"role": "user", "content": user}],
               "max_tokens": 2, "temperature": 0, "logprobs": True,
               "top_logprobs": 20, "chat_template_kwargs": {"enable_thinking": False}}

def request_hash(model, sysmsg, user):
    return hashlib.sha256(json.dumps(request_payload(model, sysmsg, user)).encode("utf-8")).hexdigest()


def call_one(base, model, sysmsg, user):
    return http_post(base, request_payload(model, sysmsg, user))

def first_tlp(resp):
    try:
        return resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        return None

def fetch_distribution(base, model, sysmsg, user, options, attempts=3, backoff=2.0):
    """Up to `attempts` calls. Retry on HTTP/JSON errors, on a missing or empty
    top_logprobs, or on an invalid extracted distribution. Returns (dist, tlp)
    on success or (None, None) after the attempts are exhausted."""
    for attempt in range(attempts):
        try:
            resp = call_one(base, model, sysmsg, user)
            tlp = first_tlp(resp)
            if not tlp:
                if attempt < attempts - 1:
                    time.sleep(backoff)
                continue
            dist, _ = build_distribution(tlp, options)
            if dist is None:
                if attempt < attempts - 1:
                    time.sleep(backoff)
                continue
            return dist, tlp
        except Exception:
            if attempt < attempts - 1:
                time.sleep(backoff)
            continue
    return None, None

def _fill(rec, dist, tlp, mapped_label, t0):
    if dist is None:
        rec.update(failed=True, error="distribution_failed")
        return
    top, top_p, margin, ent = score_dist(dist)
    rec.update(distribution=dist, top=top, top_p=top_p, margin=margin,
        entropy=ent, correct=(top == mapped_label),
        latency_ms=round((time.time() - t0) * 1000), raw_top_logprobs=tlp)

def process_A(base, model, item, ablate=False):
    try:
        sysmsg, user, options, gt, nbl = question_A(item)
    except Exception as e:
        return {"id": item.get("id"), "label": item.get("expect", "__none__"),
            "request_hash": None, "options": [], "router_persona": item.get("router_persona"),
            "expect": item.get("expect"), "failed": True,
            "error": "build_" + type(e).__name__}
    rec = {"id": item["id"], "label": gt, "options": options,
        "router_persona": item.get("router_persona"), "expect": item.get("expect")}
    rec["request_hash"] = request_hash(model, sysmsg, user)
    esc_letter = options[-1]
    t0 = time.time()
    dist, tlp = fetch_distribution(base, model, sysmsg, user, options)
    _fill(rec, dist, tlp, gt, t0)
    expect = item.get("expect", "__none__")
    rp = item.get("router_persona")
    rec["router_exact"] = (rp is None) if expect == "__none__" else (rp == expect)
    if ablate:
        rec["ablate_top"] = None
        rec["ablate_agree"] = None
        try:
            s2, u2, o2, _, nbl2 = question_A(item, reverse=True)
            rec["ablate_request_hash"] = request_hash(model, s2, u2)
            esc2 = o2[-1]
            d2, _ = fetch_distribution(base, model, s2, u2, o2)
            normal_ok = (not rec.get("failed")) and rec.get("top") is not None
            if d2 is None or not normal_ok:
                rec["ablate_top"] = None
            else:
                t2 = max(d2.items(), key=lambda kv: kv[1])[0]
                rec["ablate_top"] = t2
                name1 = "escalate" if rec["top"] == esc_letter else nbl.get(rec["top"])
                name2 = "escalate" if t2 == esc2 else nbl2.get(t2)
                rec["ablate_agree"] = (name1 == name2)
        except Exception:
            rec["ablate_top"] = None
    return rec

def process_B(base, model, item):
    try:
        sysmsg, user, options = question_B(item)
    except Exception as e:
        return {"id": item.get("id"), "label": item.get("label"), "request_hash": None, "options": [],
            "failed": True, "error": "build_" + type(e).__name__}
    rec = {"id": item["id"], "label": item["label"], "options": options}
    rec["request_hash"] = request_hash(model, sysmsg, user)
    mapped_label = LABEL_MAP.get(item["label"], item["label"])
    t0 = time.time()
    dist, tlp = fetch_distribution(base, model, sysmsg, user, options)
    _fill(rec, dist, tlp, mapped_label, t0)
    return rec

# ---- Metrics ---------------------------------------------------------------

def _mapped(label):
    return LABEL_MAP.get(label, label)

def _is_correct(r):
    """Correctness uses the mapped label everywhere so Set B (APPLY/REJECT) is
    compared against its Yes/No option space."""
    return r["top"] == _mapped(r["label"])

def metrics(records, setname):
    ok = [r for r in records if not r.get("failed") and "top_p" in r]
    n = len(ok)
    n_total, n_failed = len(records), len(records) - len(ok)
    if n == 0:
        m = {"set": setname, "n": 0, "n_total": n_total, "n_failed": n_failed,
            "accuracy": None, "majority_baseline": None, "brier": None,
            "ece": None, "ece_bins": None, "hcer": None, "hcer_n": 0,
            "coverage_5pct": None, "coverage_threshold": None, "coverage_prefix": 0}
        if setname == "A":
            m["router_exact"] = None
        else:
            m["confusion"] = None
        return m
    m = {"set": setname, "n": n, "n_total": n_total, "n_failed": n_failed,
        "accuracy": sum(1 for r in ok if _is_correct(r)) / n,
        "majority_baseline": max(Counter(_mapped(r["label"]) for r in ok).values()) / n}
    if setname == "A":
        m["router_exact"] = sum(1 for r in ok if r.get("router_exact")) / n
    # Brier: mean over items of sum over options of (p - y)^2 with one-hot y
    # built from the mapped label.
    m["brier"] = sum((r["distribution"].get(opt, 0.0)
        - (1.0 if opt == _mapped(r["label"]) else 0.0)) ** 2
        for r in ok for opt in r["options"]) / n
    # ECE 10 equal-width bins on top_p
    ece, bins = 0.0, []
    for i in range(10):
        lo, hi = i / 10.0, (i + 1) / 10.0
        memb = [r for r in ok if lo <= r["top_p"] < hi] if i < 9 else \
               [r for r in ok if lo <= r["top_p"] <= hi + 1e-9]
        if memb:
            acc_b = sum(1 for r in memb if _is_correct(r)) / len(memb)
            conf_b = sum(r["top_p"] for r in memb) / len(memb)
            ece += abs(acc_b - conf_b) * len(memb) / n
            bins.append({"lo": lo, "hi": hi, "n": len(memb), "acc": acc_b, "conf": conf_b})
    m["ece"], m["ece_bins"] = ece, bins
    # HCER: among items with top_p >= 0.90, the error rate and the count.
    hi_items = [r for r in ok if r["top_p"] >= 0.90]
    m["hcer_n"] = len(hi_items)
    m["hcer"] = (sum(1 for r in hi_items if not _is_correct(r)) / len(hi_items)) \
        if hi_items else None
    # coverage at a 5% error budget: scan ALL prefixes of the top_p-descending
    # order; the answer is the LARGEST prefix whose cumulative error rate is
    # <= 0.05; the threshold is that prefix's last top_p. 0 / null if none.
    order = sorted(ok, key=lambda r: r["top_p"], reverse=True)
    cov, thr = 0, None
    for k in range(1, n + 1):
        if sum(1 for r in order[:k] if not _is_correct(r)) / k <= 0.05:
            cov, thr = k, order[k - 1]["top_p"]
    m["coverage_5pct"], m["coverage_threshold"], m["coverage_prefix"] = cov / n, thr, cov
    if setname == "B":
        cm = {"Yes_APPLY": 0, "Yes_REJECT": 0, "No_APPLY": 0, "No_REJECT": 0}
        for r in ok:
            pred = "Yes" if r["top"] == "Yes" else "No"
            cm[f"{pred}_{r['label']}"] += 1
        m["confusion"] = cm
    return m

# ---- Summary / markdown ----------------------------------------------------

def _fmt(v, spec=".3f"):
    return "—" if v is None else format(v, spec)

def md_row(sname, m):
    rex = _fmt(m.get("router_exact")) if sname == "A" else "—"
    hcer = m.get("hcer")
    hcer_s = f"{hcer:.3f}({m.get('hcer_n', 0)})" if hcer is not None else "— (0)"
    thr = m.get("coverage_threshold")
    thr_s = f"{thr:.3f}" if thr is not None else "—"
    return (f"| {sname} | {m.get('n', 0)} | {_fmt(m.get('accuracy'))} | "
        f"{_fmt(m.get('majority_baseline'))} | {rex} | {_fmt(m.get('brier'), '.4f')} | "
        f"{_fmt(m.get('ece'), '.4f')} | {hcer_s} | "
        f"{_fmt(m.get('coverage_5pct'))} ({thr_s}) |")

def confusion_md(m):
    if m.get("confusion") is None:
        return []
    cm = m["confusion"]
    return ["", "| Set B confusion | label APPLY | label REJECT |",
            "|---|---|---|",
            f"| predicted Yes | {cm['Yes_APPLY']} | {cm['Yes_REJECT']} |",
            f"| predicted No | {cm['No_APPLY']} | {cm['No_REJECT']} |"]

def ablate_summary(recs, m):
    compared = [r for r in recs if r.get("ablate_agree") is not None]
    attempted = [r for r in recs if "ablate_top" in r]
    n_compared = len(compared)
    n_failed = len(attempted) - n_compared
    frac = (sum(1 for r in compared if r["ablate_agree"]) / n_compared) \
        if n_compared else None
    m["ablate_n_compared"] = n_compared
    m["ablate_failed"] = n_failed
    m["ablate_agreement"] = frac
    fr = f"{frac:.3f}" if frac is not None else "—"
    return (f"| A (order ablation) | agree {fr} over {n_compared} | "
            f"compared {n_compared} failed {n_failed} | | | | | | |")

# ---- Selftest --------------------------------------------------------------

def selftest(base, model, data_dir):
    all_ok = True
    for fname, proc in [("setA_routing.json", process_A), ("setB_reconcile.json", process_B)]:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"[selftest] missing {fname}"); all_ok = False; continue
        with open(path) as stream:
            items = json.load(stream)
        if not items:
            print(f"[selftest] {fname} has no items"); all_ok = False; continue
        for it in items[:2]:
            try:
                qa = question_A(it) if fname.startswith("setA") else question_B(it)
            except Exception as e:
                print(f"[selftest] {it.get('id')} build failed: {type(e).__name__}")
                all_ok = False; continue
            sysmsg, user = qa[0], qa[1]
            rec = proc(base, model, it)
            print(f"\n=== {fname} {it['id']} ===\nSYS:", sysmsg, "\nUSER:", user)
            d = rec.get("distribution")
            if d:
                print("distribution:", {k: round(v, 4) for k, v in d.items()})
                print("top:", rec.get("top"), "label:", rec.get("label"),
                      "correct:", rec.get("correct"))
            else:
                print("FAILED:", rec.get("error", "no distribution")); all_ok = False
    sys.exit(0 if all_ok else 1)

# ---- Main ------------------------------------------------------------------

def _nonneg_int(s):
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer")
    if v < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return v

def _load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def run_metrics_only(mdir, tag, setsel, model, ablate, outdir):
    summary = {"model": model, "tag": tag, "sets": {}}
    md = ["| set | n | accuracy | majority | router_exact | brier | ece | hcer(n) | coverage@5% (thr) |",
          "|---|---|---|---|---|---|---|---|---|"]
    sets = ({"A": "setA_routing.json"} if setsel == "A" else
            {"B": "setB_reconcile.json"} if setsel == "B" else
            {"A": "setA_routing.json", "B": "setB_reconcile.json"})
    for sname in sets:
        path = os.path.join(mdir, f"{sname}_{tag}.jsonl")
        if not os.path.exists(path):
            print(f"[skip] {path} missing"); continue
        recs = _load_jsonl(path)
        recs.sort(key=lambda r: r.get("id", ""))
        m = metrics(recs, sname)
        summary["sets"][sname] = m
        md.append(md_row(sname, m))
        if sname == "B":
            md.extend(confusion_md(m))
        if ablate and sname == "A":
            md.append(ablate_summary(recs, summary["sets"][sname]))
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"summary_{tag}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n".join(md) + f"\n\n[written] {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="MODEL_NAME")
    ap.add_argument("--data"); ap.add_argument("--out")
    ap.add_argument("--set", choices=["A", "B", "all"], default="all")
    ap.add_argument("--limit", type=_nonneg_int, default=None)
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--metrics-only", default=None, metavar="DIR",
        help="recompute summary/markdown from existing OUT/<set>_<tag>.jsonl "
             "files in DIR without any HTTP call")
    args = ap.parse_args()

    tag = args.tag or "".join(c if c.isalnum() else "-" for c in args.model)
    if args.selftest:
        if not args.data:
            sys.exit("--data required for --selftest")
        selftest(args.base, args.model, args.data)
    if args.metrics_only:
        if not os.path.isdir(args.metrics_only):
            sys.exit(f"--metrics-only: {args.metrics_only} is not a directory")
        if not args.out:
            sys.exit("--out required for --metrics-only")
        run_metrics_only(args.metrics_only, tag, args.set, args.model, args.ablate, args.out)
        return
    if not args.data or not args.out:
        sys.exit("--data and --out are required")

    os.makedirs(args.out, exist_ok=True)
    sets = ({"A": "setA_routing.json"} if args.set == "A" else
            {"B": "setB_reconcile.json"} if args.set == "B" else
            {"A": "setA_routing.json", "B": "setB_reconcile.json"})

    summary = {"model": args.model, "tag": tag, "sets": {}}
    md = ["| set | n | accuracy | majority | router_exact | brier | ece | hcer(n) | coverage@5% (thr) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for sname, fname in sets.items():
        path = os.path.join(args.data, fname)
        if not os.path.exists(path):
            print(f"[skip] {path} missing"); continue
        with open(path) as stream:
            items = json.load(stream)
        if args.limit is not None:
            items = items[:args.limit]
        proc, recs = (process_A if sname == "A" else process_B), [None] * len(items)
        def work(i):
            return i, (proc(args.base, args.model, items[i], ablate=args.ablate)
                       if sname == "A" else proc(args.base, args.model, items[i]))
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for f in as_completed([ex.submit(work, i) for i in range(len(items))]):
                i, r = f.result(); recs[i] = r
        recs.sort(key=lambda r: r["id"])
        with open(os.path.join(args.out, f"{sname}_{tag}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        m = metrics(recs, sname)
        summary["sets"][sname] = m
        md.append(md_row(sname, m))
        if sname == "B":
            md.extend(confusion_md(m))
        if args.ablate and sname == "A":
            md.append(ablate_summary(recs, summary["sets"][sname]))
    with open(os.path.join(args.out, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n".join(md) + f"\n\n[written] {os.path.join(args.out, f'summary_{tag}.json')}")

if __name__ == "__main__":
    main()
