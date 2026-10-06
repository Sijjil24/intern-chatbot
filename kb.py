"""Tiny knowledge-base search. Pure Python, no numpy needed."""
import json
import math
import re
from collections import Counter
from pathlib import Path

STOP = {
    "a", "an", "the", "is", "are", "to", "of", "in", "on", "for", "and", "or", "i",
    "my", "me", "do", "does", "how", "what", "can", "will", "it", "at", "be", "you",
    "your", "with", "about", "this", "that", "get", "we", "us", "am", "there", "any", "have", "has", "when", "where", "which", "who", "if", "so", "please", "tell", "need", "want",
}


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words]
    return [w for w in words if w not in STOP and len(w) > 1]


class KnowledgeBase:
    def __init__(self, path="knowledge_base/faqs.json"):
        p = Path(path)
        if not p.is_absolute():
            # Resolve relative paths from this file's folder (needed on Vercel)
            p = Path(__file__).resolve().parent / p
        self.path = p
        self.entries = json.loads(self.path.read_text(encoding="utf-8"))
        self.docs = []
        df = Counter()
        for e in self.entries:
            # question and tags count more than the answer body
            tokens = (
                tokenize(e["question"]) * 3
                + tokenize(" ".join(e.get("tags", []))) * 3
                + tokenize(e["answer"])
            )
            self.docs.append(Counter(tokens))
            df.update(set(tokens))
        n = len(self.entries)
        self.idf = {t: math.log(1 + n / c) for t, c in df.items()}

    def topics(self):
        seen, out = set(), []
        for e in self.entries:
            c = e.get("category", "General")
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def by_category(self, category):
        return [e for e in self.entries if e.get("category") == category]

    def search(self, query, k=3, min_score=0.5):
        q = tokenize(query)
        if not q:
            return []
        scored = []
        for e, doc in zip(self.entries, self.docs):
            length = sum(doc.values()) or 1
            score = sum(self.idf.get(t, 0) * doc[t] / length ** 0.5 for t in q if t in doc)
            if score >= min_score:
                scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:k]]