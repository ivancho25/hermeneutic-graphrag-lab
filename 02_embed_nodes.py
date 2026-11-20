"""
02_embed_nodes.py
Generates L2-normalized dense embeddings for all graph nodes.

Hermeneutic design choice (see data/ontology.json):
Each node's primary `text` is intentionally concatenated with its `notes`
(the researcher's situated analytical reading) prior to vectorization, steering
semantic retrieval toward argumentatively salient passages.
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMB_DIR = os.path.join(BASE_DIR, "embeddings")

# Default English sentence-transformer (~90MB, 384-dim).
# Override via EMBEDDING_MODEL env var (e.g. "paraphrase-multilingual-MiniLM-L12-v2")
# if working with multilingual corpora.
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs(EMB_DIR, exist_ok=True)

    # 1. Load all nodes in deterministic order
    all_nodes = []
    for filename in ("nodes.json", "nodes_freud.json", "nodes_theory.json"):
        data = load_json(filename)
        all_nodes.extend(data["nodes"])

    print(f"Total nodes to embed: {len(all_nodes)}")

    # 2. Prepare texts (primary passage + explicit analyst note)
    node_ids = []
    texts_to_embed = []

    for node in all_nodes:
        nid = node["id"]
        text = node["text"]
        notes = node.get("metadata", {}).get("notes", "")

        full_text = f"{text}\n\n[Analyst Note]: {notes}" if notes else text
        node_ids.append(nid)
        texts_to_embed.append(full_text)

    # 3. Load model & compute L2-normalized embeddings
    print(f"Loading embedding model: {MODEL_NAME}")
    print("(First run downloads model weights from HuggingFace and caches them locally)")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating normalized embeddings...")
    embeddings = model.encode(
        texts_to_embed,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalized -> cosine similarity == dot product
    )

    print(f"Embedding matrix shape: {embeddings.shape}")

    # 4. Save artifacts + model metadata so 03_retrieval.py stays strictly in sync
    np.save(os.path.join(EMB_DIR, "embeddings.npy"), embeddings)

    with open(os.path.join(EMB_DIR, "node_ids.json"), "w", encoding="utf-8") as f:
        json.dump(node_ids, f, ensure_ascii=False, indent=2)

    config_meta = {
        "model_name": MODEL_NAME,
        "num_nodes": len(node_ids),
        "embedding_dim": int(embeddings.shape[1]),
        "includes_analyst_notes": True,
    }
    with open(os.path.join(EMB_DIR, "embedding_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved artifacts in {EMB_DIR}/:")
    print(f"  embeddings.npy        — float32 matrix {embeddings.shape}")
    print(f"  node_ids.json         — {len(node_ids)} ordered node IDs")
    print(f"  embedding_config.json — model configuration ({MODEL_NAME})")
    print("\nEmbeddings ready.")


if __name__ == "__main__":
    main()
