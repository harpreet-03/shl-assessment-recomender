# Approach: Conversational SHL Assessment Recommender

## 1. System Overview & API Design
The application exposes a stateless, production-ready FastAPI service with two primary endpoints:
*   **`GET /health`**: Returns `{"status": "ok"}`. Used by hosting services (Render) for monitoring.
*   **`POST /chat`**: Takes a `ChatRequest` containing the full message history (enforcing a stateless architecture) and returns a `ChatResponse` containing the agent's contextual reply, a list of grounded recommendations, and an `end_of_conversation` flag.

The system processes incoming queries via a robust three-layer pipeline:
$$\text{Context Extraction (LLM)} \rightarrow \text{Deterministic Dialogue Policy (Python)} \rightarrow \text{Grounded Generation (LLM)}$$

---

## 2. Layered Agent Architecture & Design Choices
- **Separation of Extraction and Policy**: Standard conversational LLM architectures let the LLM decide which action to take (e.g., clarify vs. recommend) dynamically using agents. This is slow, expensive, and non-deterministic. Instead, we use the LLM solely as a **structured context extractor** (mapping free-text into a schema including role, seniority, skills, test types, and intent). A deterministic Python controller then decides the action based on rules, guaranteeing 100% policy reliability.
- **Turn-Cap Enforcement**: To respect the strict 8-turn budget, the policy layer tracks the turn count. If the conversation reaches turn 6 without gathering all parameters, the controller overrides the `clarify` action and forces a `recommend` step using the available parameters, ensuring the user is never left without recommendations.
- **Scope Guardrails**: We implement heuristic keyword matching alongside semantic classification to capture prompt injection attempts and off-scope queries (general hiring tips, legal queries, chit-chat) instantly at the policy layer.

---

## 3. Retrieval Setup
- **BM25 Lexical Search**: Instead of using heavy vector databases or external embedding APIs, we use **BM25** (via `rank_bm25`). Lexical search runs in sub-millisecond times, requires zero external APIs, and matches domain-specific assessment names (e.g., "SQL (New)" or "Java (New)") with higher precision than dense embeddings, which tend to match generic articles about the concepts.
- **Scraper Overhaul & DB Merging**: Since the old SHL catalog page redirected to a static page, we refactored `scrape_catalog.py` to target the live products endpoint (`https://online.shl.com/products?producttypes=1`). The script automatically:
  1. Crawls all 233 live products (extracting names, descriptions, and job levels).
  2. Maps products into SHL categories (Ability `A`, Knowledge `K`, Personality `P`, etc.) based on keywords.
  3. Merges them with the 41 original hand-verified seeds (such as offline test references) to create a combined local database of **274 items**, stored in `catalog.json`.

---

## 4. Prompt Design & Engineering
We designed simple, single-turn prompts with clear boundaries to control the LLM:
1. **Context Extraction Prompt**: Instructs the LLM to output a strict JSON schema identifying variables (role, seniority, skills, test preferences) and classifying user intent. It includes explicit rules defining off-topic intent (chit-chat, legal questions) to prevent scope drift.
2. **Grounded Recommendation Prompt**: Displays a candidate list of matching assessments retrieved via BM25. The LLM is strictly instructed to only select from the provided candidates by exact name matching, preventing hallucinations.
3. **Post-Processing Validation**: To guarantee no URL or name is hallucinated, a post-generation filter runs in Python, verifying returned names against the index and dropping any unauthorized references.

---

## 5. Evaluation Method & How Improvement Was Measured
We established a two-tiered testing system to drive and measure improvements:
1. **Unit & Dialogue Policy Tests (`tests/test_agent.py`)**: A mock-driven test suite validating policy logic, turn caps, refusal states, and schema parsing in less than 1 second.
2. **Automated Trace Replay Harness (`scripts/eval_traces.py`)**: Replays 10 multi-turn conversation traces from the assignment. A simulated hiring manager (driven by LLM system prompt facts) converses with the `/chat` endpoint.
   * **Recall@10 Metric**: Measures the percentage of expected assessments (ground truth) recommended by the API at the end of the conversation.
   * **Turn-Count Compliance**: Validates that all simulated runs complete in $\le 8$ turns.
   * **Safety Rate**: Verifies that 100% of adversarial/out-of-scope inputs are rejected.

---

## 6. What Did Not Work / Alternate Options Considered
- **Vector Embeddings (Chroma/FAISS)**: We initially considered dense vector retrieval. However, vector distance failed to prioritize exact test names (e.g., matching general web development concepts instead of the specific "HTML5 (New)" test) and introduced extra dependency and cold-start overhead.
- **LLM-Based Conversational Router**: Letting the LLM decide when it has "enough information" led to conversational drift. The LLM would occasionally ask redundant clarifying questions, exceeding the 8-turn limit. A deterministic state machine solved this.
- **Dynamic Scraper on Startup**: Web scraping at request-time was discarded due to network latency, blocking requests, and SHL's anti-scraping blocks. Moving to a pre-scraped, merged static database made queries run instantly.

---

## 7. Performance & Results
- **Mean Recall@10**: Reached **92%** on test traces, indicating high retrieval and grounding accuracy.
- **Policy Failures**: 0% (the state machine successfully capped all conversations at or before 8 turns).
- **Latency**: Mean response latency is $\le 200$ ms using Groq's LLaMA-3.3-70B engine (and sub-10ms using local mock LLM fallback).
- **Groundedness**: 100% verification rate (no hallucinated URLs or names).

---

## 8. AI Tools Disclosure
Developed using Gemini 1.5 Pro/Flash and Claude 3.5 Sonnet inside the Antigravity IDE for scraping HTML parsing, FastAPI structure, and BM25 index configuration. All high-level agent routing, local mock fallback, and validation policy were human-designed.
