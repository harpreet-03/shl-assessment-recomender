"""
eval_traces.py
--------------
Runs a Recall@10 evaluation against the assignment's 10 provided conversation
traces. This needs:
  1. A working LLM_API_KEY in .env (real calls, not stubs - unlike test_agent.py).
  2. The trace files downloaded from the assignment's "Link" and unzipped into
     tests/traces/*.json (or .yaml - adjust load_trace() below to match).

IMPORTANT: the assignment doesn't specify the exact trace file schema in the
PDF text, only that "each trace is a persona with a fact set and a labeled
expected shortlist." Once you unzip the real files, open one and adjust
`load_trace()` below to pull out:
   - persona / fact set   -> used to drive the simulated user
   - expected shortlist   -> list of assessment names or URLs, used for Recall@K

This script simulates the user with the SAME LLM (given the persona/facts as
a system prompt instructing it to answer truthfully from those facts, say "no
preference" for anything outside them, and end once a shortlist arrives -
mirroring the harness description in the assignment), then replays a real
multi-turn conversation against your running /chat endpoint.

Usage:
    python scripts/eval_traces.py --api-url http://localhost:8000 --traces-dir tests/traces
"""
import argparse
import glob
import json
import os
import sys

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
load_dotenv()
from app.llm import chat_completion  # noqa: E402

MAX_TURNS = 8

USER_SIM_SYSTEM = """You are role-playing a hiring manager with this persona and fact set:
{facts}

Rules:
- Answer the assistant's questions truthfully using ONLY these facts.
- If asked about something not in your facts, say you have no preference.
- Once the assistant gives you a shortlist of assessments, say thanks and stop (say "That works, thanks!").
- Keep replies short and natural, like a real chat message.
"""


def load_trace(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # TODO: adjust these keys once you've inspected the real downloaded trace schema.
    return {
        "persona": data.get("persona") or data.get("fact_set") or data,
        "expected": data.get("expected_shortlist") or data.get("labeled_shortlist") or data.get("expected") or [],
    }


def simulate_user_turn(facts: dict, history: list) -> str:
    sys_msg = {"role": "system", "content": USER_SIM_SYSTEM.format(facts=json.dumps(facts))}
    convo = [sys_msg] + [{"role": "assistant" if m["role"] == "assistant" else "user", "content": m["content"]}
                          for m in history]
    # flip perspective: from the simulated user's POV, the agent's turns are "assistant" already correct
    return chat_completion(convo, temperature=0.4, max_tokens=100)


def recall_at_k(expected: list, got_names: list, k: int = 10) -> float:
    if not expected:
        return 1.0
    expected_set = {e.lower() for e in expected}
    got_set = {g.lower() for g in got_names[:k]}
    hits = len(expected_set & got_set)
    return hits / len(expected_set)


def run_trace(api_url: str, trace: dict) -> float:
    history = []
    final_recs = []
    first_user_msg = simulate_user_turn(trace["persona"], history)
    history.append({"role": "user", "content": first_user_msg})

    for _ in range(MAX_TURNS):
        resp = requests.post(f"{api_url}/chat", json={"messages": history}, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        history.append({"role": "assistant", "content": body["reply"]})
        if body["recommendations"]:
            final_recs = [r["name"] for r in body["recommendations"]]
        if body["end_of_conversation"] or len(history) >= MAX_TURNS:
            break
        next_user_msg = simulate_user_turn(trace["persona"], history)
        history.append({"role": "user", "content": next_user_msg})

    return recall_at_k(trace["expected"], final_recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--traces-dir", default="tests/traces")
    args = ap.parse_args()

    trace_files = sorted(glob.glob(os.path.join(args.traces_dir, "*.json")))
    if not trace_files:
        print(f"No trace files found in {args.traces_dir}. Download & unzip the assignment's traces there first.")
        return

    scores = []
    for path in trace_files:
        trace = load_trace(path)
        score = run_trace(args.api_url, trace)
        scores.append(score)
        print(f"{os.path.basename(path)}: Recall@10 = {score:.2f}")

    print(f"\nMean Recall@10 across {len(scores)} traces: {sum(scores) / len(scores):.3f}")


if __name__ == "__main__":
    main()
