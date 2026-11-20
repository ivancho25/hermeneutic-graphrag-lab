"""
05_baseline_no_rag.py
Control baseline ("No-GraphRAG"): feeds ONLY the raw clinical narrative text
to DeepSeek without Space B, without analyst notes, without epistemic edges,
and without ontology constraints—allowing a clean A/B methodological comparison.
"""

import argparse
import json
import os
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def load_raw_story():
    path = os.path.join(BASE_DIR, "data", "nodes.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Concatenate ONLY raw paragraphs—no tags, no analyst notes, no graph
    return "\n\n".join(node["text"] for node in data["nodes"])


def run_baseline(query, temperature=0.3):
    if not DEEPSEEK_API_KEY:
        print("[Error] DEEPSEEK_API_KEY is not set in .env")
        return None

    raw_story = load_raw_story()

    system_prompt = (
        "You are a psychoanalytic and humanities research assistant. "
        "Write in lucid, analytical, essayistic prose (not bullet points). "
        "Maintain an analytical, non-prescriptive tone."
    )
    user_prompt = (
        f"Read the following first-person clinical vignette and answer the research question:\n\n"
        f"=== CLINICAL VIGNETTE ===\n{raw_story}\n\n"
        f"=== QUESTION ===\n{query}"
    )

    print(f"\n{'=' * 70}")
    print(f"[BASELINE — NO GRAPHRAG] QUERY: {query}")
    print("=" * 70)

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

    response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        print(f"API Error {response.status_code}: {response.text}")
        return None

    data = response.json()
    answer = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    print(
        f"Tokens used — prompt: {usage.get('prompt_tokens', '?')} | "
        f"completion: {usage.get('completion_tokens', '?')}\n"
    )
    print(answer)
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline (No-GraphRAG) comparison runner")
    parser.add_argument("query", help="Research question to run against the raw text only.")
    args = parser.parse_args()
    run_baseline(args.query)
