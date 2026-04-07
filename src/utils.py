import json
import re
from pathlib import Path
from typing import List


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data, path: str):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sentence_split(text: str) -> List[str]:
    parts = re.split(r"[。！？!?；;\n]", text)
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> List[str]:
    try:
        import jieba
        tokens = [w.strip() for w in jieba.lcut(text) if w.strip()]
        return tokens
    except Exception:
        return re.findall(r"[A-Za-z][A-Za-z\.\-]+|[A-Za-z]+|[\u4e00-\u9fff]{1,4}|\d{4}", text)


def text_similarity(text: str, keywords: List[str]) -> float:
    text_lower = text.lower()
    score = 0.0
    for kw in keywords:
        if kw.lower() in text_lower:
            score += 1.0
    return score / len(keywords) if keywords else 0.0


def contains_whole_word(text: str, word: str) -> bool:
    if re.search(r"[A-Za-z]", word):
        pattern = r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])"
        return re.search(pattern, text) is not None
    return word in text