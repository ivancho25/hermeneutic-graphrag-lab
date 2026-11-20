"""
03_retrieval.py
Hybrid epistemic retrieval: dense semantic search + ontology-aware graph expansion.

Key humanistic GraphRAG mechanics implemented here:
1. Dense seeds retrieve passages grounded in text + situated analyst notes.
2. Multi-hop BFS expands along explicit argumentative edges, prioritizing higher
   epistemic weights (`weight`) and optional minimum-weight thresholds.
3. Edges marked `propagate_embeddings: false` in ontology.json (e.g., OPENS_QUESTION)
   include their immediate target to surface unresolved hermeneutic tension,
   but do NOT propagate further across multi-hop chains.
4. All edges induced among the retrieved node set (both inter-space and intra-space)
   are surfaced and ranked by weight so the LLM sees the full cognitive route.
"""

import importlib.util
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# ─── Load Validated Graph from 01_build_graph.py ──────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMB_DIR = os.path.join(BASE_DIR, "embeddings")


def _import_module_from_file(module_name, filename):
    path = os.path.join(BASE_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


graph_mod = _import_module_from_file("build_graph_mod", "01_build_graph.py")
G, ONTOLOGY, _errors = graph_mod.build_graph()
if _errors:
    raise RuntimeError(f"Graph validation failed with {len(_errors)} error(s): {_errors}")

# ─── Load Embeddings & Synchronized Model ─────────────────────────────────────

embeddings = np.load(os.path.join(EMB_DIR, "embeddings.npy"))
with open(os.path.join(EMB_DIR, "node_ids.json"), "r", encoding="utf-8") as f:
    node_ids = json.load(f)

config_path = os.path.join(EMB_DIR, "embedding_config.json")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        MODEL_NAME = json.load(f)["model_name"]
else:
    MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
embedder = SentenceTransformer(MODEL_NAME)

print(f"Retrieval engine ready ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges, model='{MODEL_NAME}').")


# ─── Hybrid Retrieval Function ────────────────────────────────────────────────

def retrieve(query, top_k_semantic=5, hops=1, min_weight=0.0, verbose=True):
    """
    Hybrid retrieval: dense cosine similarity seeds + epistemic graph traversal.

    Args:
        query (str): Natural-language research or clinical question.
        top_k_semantic (int): Number of initial seed nodes via vector similarity.
        hops (int): Number of graph-expansion hops from seed nodes.
        min_weight (float): Minimum edge weight [0.0, 1.0] required to traverse an edge.
        verbose (bool): Print step-by-step trace of seeds, hops, and context stats.

    Returns:
        tuple[str, list[str]]: (formatted_context_for_llm, ordered_node_ids)
    """
    # STEP 1: Encode query & compute cosine similarity (dot product on L2-normalized vectors)
    query_vec = embedder.encode([query], normalize_embeddings=True)[0]
    similarities = embeddings @ query_vec

    top_indices = np.argsort(similarities)[::-1][:top_k_semantic]
    seed_nodes = [node_ids[i] for i in top_indices]

    if verbose:
        print(f"\n── Semantic Seed Search (top {top_k_semantic}) ──")
        for rank, nid in enumerate(seed_nodes, start=1):
            sim = similarities[id_to_idx[nid]]
            snippet = G.nodes[nid]["text"][:70].replace("\n", " ") + "..."
            print(f"  {rank}. [{sim:.3f}] {nid} ({G.nodes[nid]['space']}): {snippet}")

    # STEP 2: Epistemic Multi-Hop Expansion (BFS respecting propagate_embeddings & min_weight)
    retrieved_set = set(seed_nodes)
    # Track discovery metadata for non-seed nodes: nid -> (hop_discovered, max_incident_weight)
    expanded_meta = {}
    current_frontier = set(seed_nodes)

    for hop in range(1, hops + 1):
        next_frontier = set()

        for nid in current_frontier:
            if nid not in G:
                continue

            # Inspect both outgoing (nid -> nbr) and incoming (nbr -> nid) directed edges
            incident_neighbors = [
                (nbr, G.edges[nid, nbr]) for nbr in G.successors(nid)
            ] + [
                (nbr, G.edges[nbr, nid]) for nbr in G.predecessors(nid)
            ]

            for nbr, edata in incident_neighbors:
                w = edata["weight"]
                if w < min_weight:
                    continue

                if nbr not in retrieved_set:
                    retrieved_set.add(nbr)
                    expanded_meta[nbr] = (hop, w)
                else:
                    # Update max incident weight if already discovered at this hop
                    if nbr in expanded_meta:
                        prev_hop, prev_w = expanded_meta[nbr]
                        expanded_meta[nbr] = (prev_hop, max(prev_w, w))

                # Only continue multi-hop propagation along assertive/narrative edges;
                # hermeneutic edges (OPENS_QUESTION) surface boundary tension without chaining.
                if edata.get("propagate_embeddings", True):
                    next_frontier.add(nbr)

        current_frontier = next_frontier - (retrieved_set - set(expanded_meta.keys()))
        if not current_frontier:
            break

    # Order expanded nodes by: (hop_distance ASC, max_edge_weight DESC, semantic_sim DESC)
    expanded_sorted = sorted(
        expanded_meta.keys(),
        key=lambda n: (
            expanded_meta[n][0],
            -expanded_meta[n][1],
            -float(similarities[id_to_idx[n]]) if n in id_to_idx else 0.0,
        ),
    )

    if verbose:
        print(f"\n── Epistemic Graph Expansion (+{len(expanded_sorted)} neighbor nodes, hops={hops}) ──")
        for nid in expanded_sorted:
            hop_d, max_w = expanded_meta[nid]
            print(f"  + {nid:<10} ({G.nodes[nid]['space']} | hop={hop_d} | max_weight={max_w:.1f})")

    ordered_nodes = list(seed_nodes) + expanded_sorted

    # Collect all induced edges connecting any pair of retrieved nodes (above min_weight)
    induced_edges = []
    for src, tgt, edata in G.subgraph(ordered_nodes).edges(data=True):
        if edata["weight"] >= min_weight:
            induced_edges.append((src, tgt, edata))

    # Sort edges: inter-space bridges first, then highest epistemic weight
    induced_edges.sort(
        key=lambda item: (
            0 if item[2]["scope"] == "inter_space" else 1,
            -item[2]["weight"],
            item[2]["id"],
        )
    )

    # STEP 3: Assemble Structured Context for the LLM
    blocks = ["=== RETRIEVED PASSAGES (NODES) ===\n"]

    for nid in ordered_nodes:
        if nid not in G:
            continue
        d = G.nodes[nid]
        if nid in seed_nodes:
            origin_str = f"semantic_seed (sim={similarities[id_to_idx[nid]]:.3f})"
        else:
            hop_d, max_w = expanded_meta[nid]
            origin_str = f"graph_expanded (hop={hop_d}, edge_weight={max_w:.1f})"

        block = (
            f"[{nid} | {d['space']} | status={d['status']} | {origin_str}]\n"
            f"TEXT: {d['text']}\n"
            f"ANALYST NOTE: {d['notes']}\n"
        )
        blocks.append(block)

    if induced_edges:
        blocks.append("\n=== EPISTEMIC & NARRATIVE RELATIONS (EDGES) ===\n")
        for src, tgt, edata in induced_edges:
            block = (
                f"[{src}] --{edata['type']} (weight={edata['weight']:.1f}, "
                f"epistemic={edata['epistemic_type']})--> [{tgt}]\n"
                f"Justification: {edata['justification']}\n"
            )
            blocks.append(block)

    context = "\n".join(blocks)

    if verbose:
        n_open_q = sum(1 for _, _, e in induced_edges if e["type"] == "OPENS_QUESTION")
        n_contra = sum(1 for _, _, e in induced_edges if e["type"] == "CONTRADICTS")
        print("\n── Assembled Context Stats ──")
        print(f"  Nodes included: {len(ordered_nodes)} ({len(seed_nodes)} seeds + {len(expanded_sorted)} expanded)")
        print(f"  Edges included: {len(induced_edges)} (CONTRADICTS={n_contra}, OPENS_QUESTION={n_open_q})")
        print(f"  Context length: {len(context)} characters")

    return context, ordered_nodes


# ─── Quick Standalone Test ────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_query = "What psychic function does Argo serve in Clara's mourning for Walter?"
    ctx, nodes = retrieve(sample_query, top_k_semantic=5, hops=1)

    print("\n── Context Preview (first 1,200 chars) ──")
    print(ctx[:1200])
    print("...")
