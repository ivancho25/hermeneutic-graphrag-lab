"""
01_build_graph.py
Constructs and validates the bipartite/dual-space hermeneutic knowledge graph
from JSON sources governed by data/ontology.json.
"""

import json
import os
import networkx as nx

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_json(filename):
    """Load a UTF-8 JSON file from the data directory."""
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ontology():
    """Return the project's single-source-of-truth ontology dictionary."""
    return load_json("ontology.json")


# ─── Graph Construction ───────────────────────────────────────────────────────

def normalize_weight(raw_weight, weight_map, default=0.6):
    """
    Convert categorical ('high', 'medium', 'low') or numeric weights
    into a unified float in [0.0, 1.0] using ontology.json's weight_map.
    """
    if isinstance(raw_weight, (int, float)):
        return float(raw_weight)
    if isinstance(raw_weight, str):
        key = raw_weight.strip().lower()
        if key in weight_map:
            return float(weight_map[key])
        try:
            return float(key)
        except ValueError:
            pass
    return float(default)


def build_graph():
    """
    Build and return (G, ontology, validation_errors).

    Nodes carry: text, tags, space, doc, status, argumentative_function, notes.
    Directed edges carry: id, type, justification, weight (float), scope,
    epistemic_type, and propagate_embeddings (bool).
    """
    ontology = load_ontology()
    weight_map = ontology["metadata_fields"]["weight_map"]

    relation_specs = {
        rel["id"]: rel for rel in ontology["relation_types"]["types"]
    }
    valid_tags = {
        t["id"]
        for space_key in ("space_A", "space_B")
        for t in ontology["tags"].get(space_key, [])
    }

    G = nx.DiGraph()
    errors = []

    # 1. Load nodes across Space A (clinical) and Space B (theoretical)
    node_sources = [
        ("nodes.json",        "space_A"),
        ("nodes_freud.json",  "space_B"),
        ("nodes_theory.json", "space_B"),
    ]

    for filename, space in node_sources:
        data = load_json(filename)
        for node in data["nodes"]:
            nid = node["id"]
            if nid in G:
                errors.append(f"Duplicate node ID '{nid}' in {filename}")

            for tag in node.get("tags", []):
                if tag not in valid_tags:
                    errors.append(f"Unknown tag '{tag}' on node '{nid}' ({filename})")

            meta = node.get("metadata", {})
            G.add_node(
                nid,
                text=node["text"],
                tags=node.get("tags", []),
                space=space,
                doc=node["doc"],
                status=meta.get("status", ""),
                argumentative_function=meta.get("argumentative_function", ""),
                notes=meta.get("notes", ""),
            )

    # Helper to register and validate edges consistently
    existing_nodes = set(G.nodes())

    def register_edge(edge, scope):
        eid = edge["id"]
        src = edge["source"]
        tgt = edge["target"]
        rel_type = edge["type"]

        if src not in existing_nodes:
            errors.append(f"Missing source node '{src}' on edge '{eid}' ({scope})")
            return
        if tgt not in existing_nodes:
            errors.append(f"Missing target node '{tgt}' on edge '{eid}' ({scope})")
            return
        if rel_type not in relation_specs:
            errors.append(f"Unknown relation type '{rel_type}' on edge '{eid}'")
            return
        if G.has_edge(src, tgt):
            prev_id = G.edges[src, tgt]["id"]
            errors.append(
                f"Duplicate directed edge ({src} -> {tgt}): '{eid}' overwrites '{prev_id}'"
            )

        spec = relation_specs[rel_type]
        weight = normalize_weight(
            edge.get("weight"),
            weight_map,
            default=spec.get("default_weight", 0.6),
        )

        G.add_edge(
            src,
            tgt,
            id=eid,
            type=rel_type,
            justification=edge["justification"],
            weight=weight,
            raw_weight=edge.get("weight"),
            scope=scope,
            epistemic_type=spec["epistemic_type"],
            propagate_embeddings=spec["propagate_embeddings"],
        )

    # 2. Load inter-space edges (Space A -> Space B)
    inter_edges = load_json("edges.json")["edges"]
    for edge in inter_edges:
        register_edge(edge, scope="inter_space")

    # 3. Load intra-space edges (Space A -> Space A, Space B -> Space B)
    intra_data = load_json("edges_intra.json")["edges_intra"]
    for edge in intra_data.get("space_A", []):
        register_edge(edge, scope="intra_space_A")
    for edge in intra_data.get("space_B", []):
        register_edge(edge, scope="intra_space_B")

    return G, ontology, errors


# ─── CLI Validation & Summary Report ──────────────────────────────────────────

if __name__ == "__main__":
    G, ontology, errors = build_graph()

    n_space_a = sum(1 for _, d in G.nodes(data=True) if d["space"] == "space_A")
    n_space_b = sum(1 for _, d in G.nodes(data=True) if d["space"] == "space_B")
    n_inter = sum(1 for _, _, d in G.edges(data=True) if d["scope"] == "inter_space")
    n_intra = G.number_of_edges() - n_inter

    print("── Graph Construction Summary ──")
    print(f"Total nodes loaded:       {G.number_of_nodes()} (Space A: {n_space_a} | Space B: {n_space_b})")
    print(f"Inter-space edges loaded: {n_inter}")
    print(f"Intra-space edges loaded: {n_intra}")
    print(f"Total directed edges:     {G.number_of_edges()}")

    print("\n── Ontology & Integrity Validation ──")
    if errors:
        print(f"Validation issues found ({len(errors)}):")
        for err in errors:
            print(f"  [!] {err}")
    else:
        print("OK — all node references, tags, relation types, and directed pairs are valid.")

    # Edge breakdown by relation type and epistemic function
    edge_counts = {}
    for _, _, d in G.edges(data=True):
        key = (d["type"], d["epistemic_type"])
        edge_counts[key] = edge_counts.get(key, 0) + 1

    print("\n── Edges by Relation Type ──")
    for (rel_type, epist), count in sorted(edge_counts.items()):
        print(f"  {rel_type:<18} [{epist:<17}]: {count}")

    # Most connected nodes (epistemic hubs)
    print("\n── Top 5 Epistemic Hubs (Total Degree: In + Out) ──")
    top_hubs = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]
    for nid, degree in top_hubs:
        snippet = G.nodes[nid]["text"][:65].replace("\n", " ") + "..."
        print(f"  {nid:<10} (deg {degree:>2} | {G.nodes[nid]['space']}): {snippet}")

    print("\nGraph built and verified.")
