from __future__ import annotations

import argparse
import json
from pathlib import Path

from kg_builder import KnowledgeGraphPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a knowledge graph from a txt file.")
    parser.add_argument("--input", required=True, help="Input txt file path")
    parser.add_argument("--output", default="output/kg.json", help="Output json path")
    parser.add_argument("--use-crf", action="store_true", help="Enable CRF NER if model exists")
    parser.add_argument("--crf-model", default="models/crf_ner.pkl", help="CRF model path")
    parser.add_argument(
        "--transformer-model",
        default=None,
        help="Optional local Hugging Face token classification model path/name",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = KnowledgeGraphPipeline(
        use_crf=args.use_crf,
        crf_model_path=args.crf_model,
        transformer_model=args.transformer_model,
    )
    graph = pipeline.build_from_file(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Knowledge graph saved to {output_path}")


if __name__ == "__main__":
    main()
