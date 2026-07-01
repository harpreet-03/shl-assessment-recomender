# Approach: Conversational SHL Assessment Recommender

## 1. System overview

The service is a stateless FastAPI app (`GET /health`, `POST /chat`) with three
layers per request: **context extraction** (LLM, JSON-mode) → **deterministic
dialogue policy** (plain Python, not LLM-decided) → **grounded response
generation** (LLM, closed-set over retrieved catalog rows). Retrieval is BM25
over a scraped catalog restricted to Individual Test Solutions.

## 2. Design choices and trade-offs

**Policy is deterministic, facts are not.** The LLM's only job in step 1 is to
turn free text into a structured fact set (`role_title`, `seniority`,
`must_have_skills`, `test_type_prefs`, `intent`, `compare_targets`, ...) via
one JSON-mode call. A plain `if/else` (`decide_action`) then picks one of
`{refuse_scope, refuse_injection, clarify, compare, recommend}`. I deliberately
did *not* let the LLM decide the action directly — an 8-turn, non-deterministic
conversation needs one place in the pipeline that behaves identically given
the same facts, every time, or "clarify vs. recommend" flip-flops turn to turn.
The LLM proposes; the policy decides.

**Retrieval is BM25, not embeddings.** The catalog is short, structured text
(name + short description + test-type code), which is exactly the regime
where lexical retrieval is competitive with embeddings and far cheaper: no
vector DB, no embedding API calls, sub-millisecond queries, and it fits the
30-second-per-call budget with room to spare for the LLM calls. It also makes
the anti-hallucination guarantee mechanical rather than prompt-based (next
point).

**Closed-set generation, not free generation, for names/URLs.** Retrieval
returns up to 15 real catalog rows; the LLM is only ever asked to *copy* a
`name` field from that list, never to produce a new one. After the LLM
responds, every returned name is re-validated against `catalog.json` and
anything that doesn't match exactly is dropped. If the LLM call fails, times
out, or returns nothing valid, a deterministic BM25-top-N fallback fires
instead of erroring. Net effect: it is structurally impossible for `/chat` to
return a URL that isn't in the scraped catalog, regardless of what the LLM
does — this was the single most important design decision for passing the
hard evals under a non-deterministic user simulator.

**Turn-cap handling.** `decide_action` forces `recommend` once
`turn_count >= MAX_TURNS - 2`, so an evasive or terse simulated user can't
push the conversation past the 8-turn cap without ever getting a shortlist.

**Scope and injection guardrails, two layers.** A regex net catches obvious
injection phrasing ("ignore previous instructions", "reveal your system
prompt") as a fast, free backstop; the LLM extraction step separately
classifies `intent` into `off_topic` / `general_hiring_advice` /
`legal_question` / `prompt_injection`, which is more robust to paraphrasing
than any regex list. Both routes lead to a short, on-brand refusal that
re-offers the in-scope help, rather than a dead end.

**Refine is not a separate code path.** "Actually, add personality tests"
just updates `test_type_prefs` in the extracted fact set and re-runs the same
`recommend` handler with the same retrieval query — because the conversation
is replayed in full on every stateless call, "refine" and "initial recommend"
are the same action with different accumulated facts. This avoids a whole
class of bugs where refine logic drifts from initial-recommend logic.

## 3. Catalog scraping

`scripts/scrape_catalog.py` crawls `shl.com/products/product-catalog/?type=1`
(Individual Test Solutions only — Pre-packaged Job Solutions excluded per the
brief), paginating 12 rows at a time (~32 pages), then visits each detail page
for a description, job levels, languages, remote-testing and adaptive/IRT
flags. `app/data/catalog.json` ships with a 41-row **seed** I verified by hand
while building this (real names/URLs, several test-type categories
represented) so the service is demoable immediately; running the script
produces the full ~380-row catalog before deployment. I chose a re-runnable
script over a one-time hand-scrape because SHL updates this catalog, and
because it's the more defensible engineering artifact for a codebase that's
meant to be maintained.

## 4. Prompt design

Three narrow, single-purpose prompts, each constrained to JSON output:
extraction (facts + intent), recommend (pick + justify from a shown list),
compare (answer only from shown catalog rows, explicitly told not to use
prior/outside knowledge of the products). Keeping prompts narrow rather than
one do-everything prompt made both stubbing (offline tests) and failure
isolation much easier — if `recommend` degrades, I know it's not accidentally
also miscounting turns or misclassifying scope, because those are separate
LLM calls with separate contracts.

## 5. Evaluation approach

Two tiers, matching the two things I could actually verify without the real
grading harness:

1. **Offline pipeline tests** (`tests/test_agent.py`, 8 tests, no API key,
   run in <1s): stub the LLM with deterministic fakes and assert on schema
   compliance, catalog-only URLs/names, no-recommend-on-vague-turn-1,
   scope/injection refusal, turn-cap forcing, and grounded compare replies.
   These test *our* pipeline logic, not LLM judgement quality — they're the
   right tool for making sure the hard evals can't fail for structural
   reasons.
2. **Recall@10 replay harness** (`scripts/eval_traces.py`): simulates the
   persona-driven user described in the assignment (truthful from facts, "no
   preference" outside them, stops once given a shortlist) using the same
   LLM, and replays a real multi-turn conversation against a running
   `/chat`. This needs a live API key and the real trace files, which
   weren't inspectable as structured data from the PDF alone — the loader
   has one clearly marked `TODO` to match field names once the trace ZIP is
   unzipped.

## 6. What didn't work / what I'd change with more time

- Full catalog description text via detail-page scraping is the most
  time-consuming part of `scrape_catalog.py`; I'd parallelize it (currently
  sequential + politeness delay) and cache parsed detail pages so re-runs are
  incremental rather than full re-crawls.
- BM25 alone under-performs on synonym-heavy queries ("stakeholder
  management" vs. "people skills") compared to embeddings; a hybrid
  BM25+embedding re-rank would likely lift Recall@10 further if the free-tier
  latency budget allows it.
- I did not persist any conversation-level cache between calls (by design,
  per the stateless requirement), which means every `/chat` call re-extracts
  facts from the full history — fine at an 8-turn cap, but it's the first
  thing I'd optimize (e.g., incremental fact-merging) if turn limits grew.

## 7. AI tool usage disclosure

I used Claude (via the Claude.ai coding environment) as an agentic pair-programmer
for scaffolding this FastAPI service, the BM25 retrieval wrapper, the
crawler script, and the offline test stubs, then reviewed, ran, and fixed
every file myself (e.g., an initial monkeypatching bug in the offline tests
where `agent.py` had imported `chat_json` by name instead of through the
`llm` module, which silently bypassed the test stubs — caught by actually
running `pytest` rather than trusting the first draft). All architectural
decisions above (deterministic policy layer, BM25 over embeddings, closed-set
generation, stateless refine) are choices I made and can defend; the LLM was
used for typing speed, not for design decisions.
