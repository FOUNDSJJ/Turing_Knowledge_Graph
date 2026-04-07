import json
from src.utils import load_json, save_json
from src.pipeline import KnowledgeGraphPipeline


def main():
    text = """
    Alan Turing 是英国数学家。
    Alan Turing 提出了 Turing machine。
    Alan Turing 在 Bletchley Park 参与破译 Enigma。
    Alan Turing 后来在 University of Manchester 工作。
    A.M. Turing Award 以 Alan Turing 命名。
    """

    alias_map = load_json("data/alias_map.json")
    entity_kb = load_json("data/entity_kb.json")
    seed_data = load_json("data/seed_terms.json")

    pipeline = KnowledgeGraphPipeline(alias_map, entity_kb)
    result = pipeline.build(text, seed_entities=seed_data["default_seeds"])

    # print(json.dumps(result, ensure_ascii=False, indent=2))
    save_json(result, "output_turing_kg.json")


if __name__ == "__main__":
    main()