import json
from src.utils import load_json, save_json
from src.pipeline import KnowledgeGraphPipeline


def main():
    text = """
    Alan Turing是英国数学家。
    Alan Turing提出了Turing machine。
    Alan Turing在Bletchley Park参与破译Enigma。
    Alan Turing后来在University of Manchester工作。
    A.M. Turing Award以Alan Turing命名。
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