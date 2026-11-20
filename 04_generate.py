"""
04_generate.py
End-to-end hermeneutic GraphRAG pipeline:
Query -> Hybrid Epistemic Retrieval -> Analytical Synthesis via DeepSeek API.

Reads working hypothesis and hermeneutic constraints directly from ontology.json.
"""

import argparse
import importlib.util
import os
import sys
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional if DEEPSEEK_API_KEY is exported in shell

# ─── Import Retrieval Engine from 03_retrieval.py ─────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _import_module_from_file(module_name, filename):
    path = os.path.join(BASE_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


retrieval_mod = _import_module_from_file("retrieval_mod", "03_retrieval.py")
retrieve = retrieval_mod.retrieve
ONTOLOGY = retrieval_mod.ONTOLOGY

# ─── DeepSeek API Configuration ───────────────────────────────────────────────

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # DeepSeek-V3


def build_system_prompt():
    """
    Construct the analytical system prompt, dynamically injecting the working
    hypothesis and hermeneutic edge rules from data/ontology.json.
    """
    hyp = ONTOLOGY.get("central_hypothesis", {})
    hyp_statement = hyp.get("statement", "")
    hyp_instruction = hyp.get("instruction", "")

    opens_q_rule = ""
    for rel in ONTOLOGY.get("relation_types", {}).get("types", []):
        if rel["id"] == "OPENS_QUESTION":
            opens_q_rule = rel.get("special_handling", "")
            break

    return f"""You are a psychoanalytic and humanities research assistant working over a curated dual-space epistemic corpus.

Narrative reference date: October 2024

CASE DRAMATIS PERSONAE (read carefully before interpreting any passage):
- Clara: First-person narrator, British freelance archivist, 52 years old in 2019 (Bristol), 57 years old in October 2024 (Dartmoor, Devon).
- Walter: Clara's widowed FATHER (84 in 2019). Frail and dependent on daily care in Bristol; died at home of winter pneumonia in January 2022.
- Argo: Clara's Lurcher (greyhound-collie cross, ~20 kg), adopted as a puppy in 2019; 5 going on 6 years old in October 2024.
- Julian: Clara's former partner (four-year relationship ended in 2019; brief failed reconciliation in 2022).
- Moss: Semi-feral male dog on the Dartmoor smallholding; died after eating poisoned meat bait laid by a neighbouring farmer.
- Fern: Seven-to-eight-month-old female pup belonging to the host family on Dartmoor, approaching her first season in 2024.

CENTRAL WORKING HYPOTHESIS OF THE RESEARCH:
"{hyp_statement}"
 Epistemic instruction: {hyp_instruction}

HERMENEUTIC & STYLISTIC GUIDELINES:
1. Write in lucid, rigorous, essayistic prose—like a psychoanalytic case discussion or humanities seminar paper, never a bulleted clinical checklist.
2. Never mention "the graph", node IDs (e.g. `argo_26`), or raw edge labels (`--APPLIES_CONCEPT-->`) in your prose. Weave passages and theoretical authors (Freud, Winnicott, Kohut, Bowlby, Lacan, Marie Bonaparte, Doka, Klass et al.) organically into the argument.
3. Respect edge weights: higher-weight relations (`weight=1.0`) form the spine of your interpretation; lower-weight relations (`weight=0.6` or `0.3`) introduce secondary nuances or peripheral hypotheses.
4. When `CONTRADICTS` or `NUANCES` relations appear, stage the conceptual friction explicitly rather than smoothing it over.
5. Special rule for `OPENS_QUESTION`: {opens_q_rule}
6. Maintain an analytical, non-prescriptive tone—do not offer self-help advice to Clara."""


# ─── Generation Pipeline ──────────────────────────────────────────────────────

def generate(query, top_k=5, hops=1, min_weight=0.0, temperature=0.3):
    """
    Full pipeline: Query -> Hybrid Epistemic Retrieval -> DeepSeek Synthesis.
    """
    if not DEEPSEEK_API_KEY:
        print(
            "\n[Error] DEEPSEEK_API_KEY is not set.\n"
            "Export it in your shell or create a .env file (see .env.example)."
        )
        return None

    print(f"\n{'=' * 70}")
    print(f"QUERY: {query}")
    print("=" * 70)

    context, used_nodes = retrieve(
        query,
        top_k_semantic=top_k,
        hops=hops,
        min_weight=min_weight,
        verbose=True,
    )

    system_prompt = build_system_prompt()
    user_prompt = (
        "Examine the following research question strictly through the retrieved "
        "clinical passages and theoretical relations provided below:\n\n"
        f"QUESTION: {query}\n\n"
        f"{context}\n\n"
        "Compose a cohesive analytical essay grounded in these passages and their "
        "epistemic tensions."
    )

    print(f"\n── Dispatching to DeepSeek ({DEEPSEEK_MODEL}, temp={temperature}) ──")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 4000,
        "temperature": temperature,
    }

    try:
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=120)
    except requests.RequestException as exc:
        print(f"Network error calling DeepSeek API: {exc}")
        return None

    if response.status_code != 200:
        print(f"API Error {response.status_code}: {response.text}")
        return None

    data = response.json()
    answer = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    print(
        f"Tokens used — prompt: {usage.get('prompt_tokens', '?')} | "
        f"completion: {usage.get('completion_tokens', '?')}"
    )
    print(f"\n{'=' * 70}")
    print("ANALYTICAL SYNTHESIS:")
    print("=" * 70)
    print(answer)

    return answer


# ─── CLI & Interactive Loop ───────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hermeneutic GraphRAG — Query & Analytical Synthesis"
    )
    parser.add_argument("query", nargs="?", help="Optional single question to run and exit.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of semantic seed nodes (default: 5).")
    parser.add_argument("--hops", type=int, default=1, help="Graph expansion hops (default: 1).")
    parser.add_argument("--min-weight", type=float, default=0.0, help="Minimum edge weight [0.0-1.0] (default: 0.0).")
    args = parser.parse_args()

    if args.query:
        generate(args.query, top_k=args.top_k, hops=args.hops, min_weight=args.min_weight)
        sys.exit(0)

    print("\nHermeneutic GraphRAG — Interactive Research Console")
    print("Type 'exit' or 'q' to quit.\n")

    while True:
        try:
            q = input("\nResearch Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if q.lower() in {"exit", "quit", "q", "salir"}:
            break
        if not q:
            continue
        generate(q, top_k=args.top_k, hops=args.hops, min_weight=args.min_weight)
