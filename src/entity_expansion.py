from collections import Counter
from typing import List, Tuple
from .utils import sentence_split, tokenize


class SimpleEntityExpander:
    def __init__(self, stopwords=None):
        self.stopwords = stopwords or {
            "的", "了", "和", "与", "及", "并", "是", "在", "对", "中", "研究", "进行",
            "提出", "推动", "参与", "工作", "使用", "设计", "这一", "一种", "成为",
            "重要", "基础", "相关", "系统", "通信系统", "德国", "期间", "早期", "从事"
        }

        self.bad_english_words = {
            "machine", "important", "during", "research", "use", "work", "design"
        }

    def is_valid_candidate(self, word: str) -> bool:
        if not word:
            return False
        if word in self.stopwords:
            return False
        if word.isdigit():
            return False
        if len(word) <= 1:
            return False
        if word.lower() in self.bad_english_words:
            return False
        if word.endswith("年"):
            return False

        # 纯年份不要
        if word.isdigit() and len(word) == 4:
            return False

        # 单个普通英文词尽量不要
        if word.isascii() and " " not in word and word.lower() not in {"enigma", "bombe", "turing"}:
            return False

        return True

    def expand(self, text: str, seed_entities: List[str], top_k: int = 5) -> List[Tuple[str, int]]:
        counter = Counter()
        sentences = sentence_split(text)

        for sent in sentences:
            if any(seed in sent for seed in seed_entities):
                words = tokenize(sent)
                for w in words:
                    w = w.strip()
                    if not self.is_valid_candidate(w):
                        continue
                    if w in seed_entities:
                        continue
                    counter[w] += 1

        return counter.most_common(top_k)