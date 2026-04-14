from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def sent2features(sentence: str, index: int) -> dict[str, str]:
    char = sentence[index]
    prev_char = sentence[index - 1] if index > 0 else "<BOS>"
    next_char = sentence[index + 1] if index < len(sentence) - 1 else "<EOS>"
    return {
        "char": char,
        "prev_char": prev_char,
        "next_char": next_char,
        "char.isdigit": str(char.isdigit()),
        "char.istitle": str(char.istitle()),
        "bigram_left": prev_char + char,
        "bigram_right": char + next_char,
    }


def load_bio_jsonl(path: str | Path):
    sentences = []
    tags = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            sentence = item["text"]
            label_seq = item["labels"]
            if len(sentence) != len(label_seq):
                raise ValueError("Each sentence must align with one label per character.")
            sentences.append([sent2features(sentence, idx) for idx in range(len(sentence))])
            tags.append(label_seq)
    return sentences, tags


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CRF NER model.")
    parser.add_argument("--train", required=True, help="BIO JSONL training file path")
    parser.add_argument("--output", required=True, help="Output pickle model path")
    args = parser.parse_args()

    try:
        import sklearn_crfsuite
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: sklearn-crfsuite. Install it with `pip install sklearn-crfsuite`."
        ) from exc

    x_train, y_train = load_bio_jsonl(args.train)
    model = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )
    model.fit(x_train, y_train)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(model, file)
    print(f"CRF model saved to {output_path}")


if __name__ == "__main__":
    main()
