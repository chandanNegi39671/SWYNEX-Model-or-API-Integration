"""Wrapper around the Task 1 model (word+char TF-IDF -> LinearSVC, temperature-scaled, confidence-gated)."""
from __future__ import annotations
import os, time
from pathlib import Path
import joblib
from scipy.special import softmax

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parents[1] / "models" / "model.joblib"))

class IntentRouter:
    def __init__(self, path: Path = MODEL_PATH):
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found at {path}. See README 'Model file' section.")
        b = joblib.load(path)
        self.model, self.classes, self.T, self.threshold = b["model"], list(b["classes"]), float(b["T"]), float(b["threshold"])
        self.version = f"task1-svc-wordchar-T{self.T:.4f}-thr{self.threshold}"
        self.route(["warm up"])  # first sklearn call is slow; pay it at startup, not on the first user request

    def route(self, texts: list[str], top_k: int = 3) -> list[dict]:
        P = softmax(self.model.decision_function(texts) / self.T, axis=1)
        out = []
        for p in P:
            idx = p.argsort()[::-1][:top_k]
            top = [{"intent": self.classes[i], "confidence": round(float(p[i]), 4)} for i in idx]
            out.append({"intent": top[0]["intent"], "confidence": top[0]["confidence"], "top_k": top,
                        "decision": "AUTO_ROUTE" if top[0]["confidence"] >= self.threshold else "HUMAN_REVIEW"})
        return out
