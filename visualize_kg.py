import json
from pyvis.network import Network


def load_kg(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_node_color(entity_type: str) -> str:
    color_map = {
        "PER": "#ff6b6b",
        "ORG": "#4dabf7",
        "LOC": "#51cf66",
        "WORK": "#f59f00",
        "FIELD": "#845ef7",
        "EVENT": "#20c997",
        "AWARD": "#fcc419",
        "TIME": "#adb5bd",
        "TERM": "#ced4da"
    }
    return color_map.get(entity_type, "#dee2e6")


def visualize_kg(json_path: str, output_html: str = "kg_graph.html"):
    data = load_kg(json_path)

    net = Network(
        height="750px",
        width="100%",
        bgcolor="white",
        font_color="black",
        directed=True
    )

    net.barnes_hut()

    # 添加节点
    for entity in data.get("entities", []):
        node_id = entity.get("normalized_name") or entity.get("name")
        label = entity.get("name", node_id)
        entity_type = entity.get("entity_type", "TERM")
        desc = entity.get("description", "")
        kb_id = entity.get("kb_id", "")
        aliases = entity.get("aliases", [])

        title = f"""
        <b>{label}</b><br>
        类型: {entity_type}<br>
        规范名: {node_id}<br>
        KB ID: {kb_id}<br>
        别名: {', '.join(aliases) if aliases else '无'}<br>
        描述: {desc if desc else '无'}
        """

        net.add_node(
            node_id,
            label=label,
            title=title,
            color=get_node_color(entity_type)
        )

    # 添加边
    for rel in data.get("relations", []):
        subj = rel.get("subject")
        obj = rel.get("object")
        pred = rel.get("predicate", "related_to")
        evidence = rel.get("evidence", "")

        if subj and obj:
            net.add_edge(
                subj,
                obj,
                label=pred,
                title=evidence
            )

    # 不要用 net.show()
    net.write_html(output_html, open_browser=False)
    print(f"图谱已保存到: {output_html}")


if __name__ == "__main__":
    visualize_kg("output_turing_kg.json", "turing_kg.html")