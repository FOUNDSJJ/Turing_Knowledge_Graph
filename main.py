from __future__ import annotations

import argparse
import json
from pathlib import Path

from kg_builder import KnowledgeGraphPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a knowledge graph from plain text with configurable NER and linking resources."
    )
    parser.add_argument("--input", required=True, help="Input txt file path")
    parser.add_argument("--output", default="output/kg.json", help="Output json path")
    parser.add_argument("--use-crf", action="store_true", help="Enable CRF NER if model exists")
    parser.add_argument("--crf-model", default="models/crf_ner.pkl", help="CRF model path")
    parser.add_argument(
        "--transformer-model",
        default=None,
        help="Optional local Hugging Face token classification model path or model name",
    )
    parser.add_argument(
        "--transformer-tokenizer",
        default=None,
        help="Optional tokenizer path or name when it differs from the transformer model",
    )
    parser.add_argument(
        "--config-dir",
        default="data/config",
        help="Directory containing configurable dictionaries and extraction rules",
    )
    parser.add_argument("--entity-patterns", default=None, help="Override entity pattern JSON path")
    parser.add_argument("--lexicon", default=None, help="Override lexicon JSON path")
    parser.add_argument("--alias-table", default=None, help="Override alias table JSON path")
    parser.add_argument("--knowledge-base", default=None, help="Override knowledge base JSON path")
    parser.add_argument("--relation-rules", default=None, help="Override relation rules JSON path")
    parser.add_argument(
        "--normalization-config",
        default=None,
        help="Override normalization JSON path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = KnowledgeGraphPipeline(
        use_crf=args.use_crf,
        crf_model_path=args.crf_model,
        transformer_model=args.transformer_model,
        transformer_tokenizer=args.transformer_tokenizer,
        config_dir=args.config_dir,
        entity_patterns_path=args.entity_patterns,
        lexicon_path=args.lexicon,
        alias_table_path=args.alias_table,
        knowledge_base_path=args.knowledge_base,
        relation_rules_path=args.relation_rules,
        normalization_path=args.normalization_config,
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
