from __future__ import annotations

import argparse

from kg_builder.visualization import visualize_knowledge_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize a knowledge graph JSON file as SVG or HTML."
    )
    parser.add_argument(
        "--input",
        default="output/kg_re.json",
        help="Input knowledge graph JSON path",
    )
    parser.add_argument(
        "--output",
        default="output/kg_re.svg",
        help="Output visualization path. Use .svg or .html suffix.",
    )
    parser.add_argument(
        "--hide-isolated",
        action="store_true",
        help="Only draw entities that participate in at least one relation.",
    )
    parser.add_argument(
        "--show-isolated",
        action="store_true",
        help="Draw all entities, including entities with no extracted relations.",
    )
    parser.add_argument("--width", type=int, default=1280, help="SVG canvas width")
    parser.add_argument("--height", type=int, default=860, help="SVG canvas height")
    parser.add_argument("--seed", type=int, default=42, help="Layout random seed")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = visualize_knowledge_graph(
        input_path=args.input,
        output_path=args.output,
        hide_isolated=args.hide_isolated or not args.show_isolated,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    print(
        "Knowledge graph visualization saved to "
        f"{result['output']} ({result['node_count']} nodes, {result['edge_count']} edges)"
    )


if __name__ == "__main__":
    main()
