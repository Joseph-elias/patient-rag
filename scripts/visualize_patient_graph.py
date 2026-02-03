import json
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt


GRAPH_PATH = "data/graph/patient_graph.json"
OUT_DIR = Path("assets")
OUT_DIR.mkdir(exist_ok=True)


def visualize_patient(patient_id: str):
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        graph = json.load(f)

    patient = graph["patients"].get(patient_id)
    if not patient:
        print(f"Patient {patient_id} not found.")
        return

    G = nx.DiGraph()

    pid_node = f"Patient {patient_id}"
    G.add_node(pid_node, type="patient")

    # Diagnosis
    if patient.get("diagnosis"):
        d = patient["diagnosis"]["value"]
        G.add_node(d, type="diagnosis")
        G.add_edge(pid_node, d, label="HAS_DIAGNOSIS")

    # Treatment
    if patient.get("treatment"):
        t = patient["treatment"]["value"]
        G.add_node(t, type="treatment")
        G.add_edge(pid_node, t, label="RECEIVED")

    # Adverse events
    for ev in patient.get("adverse_events", []):
        name = f"{ev['name']} (grade {ev['grade']})"
        G.add_node(name, type="adverse")
        G.add_edge(pid_node, name, label="EXPERIENCED")

    # Negated findings
    for neg in patient.get("negated_findings", []):
        name = f"NO {neg['name']}"
        G.add_node(name, type="negated")
        G.add_edge(pid_node, name, label="NEGATED")

    # Follow-up
    if patient.get("follow_up"):
        fu = patient["follow_up"]["value"]
        G.add_node(fu, type="followup")
        G.add_edge(pid_node, fu, label="FOLLOW_UP")

    # Layout
    pos = nx.spring_layout(G, seed=42)

    # Colors
    color_map = []
    for node in G.nodes(data=True):
        t = node[1].get("type")
        color_map.append({
            "patient": "#FFD700",
            "diagnosis": "#FF9999",
            "treatment": "#99CCFF",
            "adverse": "#FFCC99",
            "negated": "#CCCCCC",
            "followup": "#99FF99",
        }.get(t, "#FFFFFF"))

    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=color_map,
        node_size=2200,
        font_size=9,
        edge_color="gray"
    )

    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    out_path = OUT_DIR / f"patient_{patient_id}_graph.png"
    plt.title(f"Patient {patient_id} – Knowledge Graph")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved graph to {out_path}")


if __name__ == "__main__":
    visualize_patient("P005") 
