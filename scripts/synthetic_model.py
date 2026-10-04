"""Deliberately programmed stubs; these are not trained models or performance evidence."""
from battery import QUESTIONS, TEMPLATE


def predict(state, mode="content"):
    note = "\n".join(line for line in state.splitlines() if not line.startswith("goal:")).lower()
    goal = state.splitlines()[0].lower()
    content = "fold new entries into changelog.md and run the structure check" in note
    template = "changelog-merge" in goal if mode == "name" else content
    flags = {
        "writes": content or "modify" in note or "rotate" in note,
        "deterministic": content or "structure audit" in note,
        "egress": "publish" in note, "prod_shared": "credentials" in note or "production" in note,
        "judgment": "design" in note or "review" in note,
        "gpu_long": "train" in note, TEMPLATE: template,
    }
    return {q: 0.99 if flags[q] else 0.01 for q in QUESTIONS}
