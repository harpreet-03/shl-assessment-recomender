# SHL Conversational Assessment Recommender

Take-home submission for the SHL AI Research Intern assignment.

## 1. Project layout

```
shl-assessment-recommender/
├── app/
│   ├── main.py          FastAPI app: GET /health, POST /chat
│   ├── agent.py         Dialogue policy: clarify / recommend / refine / compare / refuse
│   ├── catalog.py        BM25 retrieval over the scraped catalog
│   ├── llm.py            Thin OpenAI-compatible LLM client (Groq by default)
│   ├── models.py          Pydantic request/response schemas (matches spec exactly)
│   └── data/catalog.json  Seed catalog (real rows) - replace with full crawl, see step 2
├── scripts/
│   ├── scrape_catalog.py  Crawler -> rebuilds app/data/catalog.json from shl.com
│   └── eval_traces.py     Recall@10 harness against the assignment's real traces
├── tests/
│   ├── test_agent.py      Offline hard-eval + behavior-probe tests (no API key needed)
│   └── traces/            Put the assignment's 10 downloaded trace files here
├── requirements.txt
├── Dockerfile
├── render.yaml
├── .env.example
└── APPROACH.md            The 2-page approach doc for submission
```

## 2. One-time setup

```bash
cd shl-assessment-recommender
python -m venv venv && source venv/bin/activate      # or use your existing env
pip install -r requirements.txt
cp .env.example .env                                  # then edit .env, see step 3
```

### Rebuild the full catalog (important - do this before submitting)

The shipped `app/data/catalog.json` is a **41-item seed** (verified real names/URLs I
pulled directly from shl.com while building this). The assignment wants "the entire
SHL catalogue" of Individual Test Solutions (~380 items across 32 listing pages). Run:

```bash
python scripts/scrape_catalog.py --out app/data/catalog.json
```

This takes a few minutes (1 request/sec, politeness delay) because it also visits each
item's detail page for a real description. For a quick smoke-test run first:

```bash
python scripts/scrape_catalog.py --out app/data/catalog.json --max-pages 3 --skip-details
```

## 3. Get a free LLM key

The agent uses one LLM for two jobs: extracting structured facts from the conversation,
and writing grounded replies. Pick one (both have generous free tiers):

- **Groq** (recommended, fast): https://console.groq.com/keys -> put the key in `.env` as `LLM_API_KEY`.
  Default `.env.example` is already pointed at Groq (`llama-3.3-70b-versatile`).
- **OpenRouter**: https://openrouter.ai/keys -> set `LLM_BASE_URL=https://openrouter.ai/api/v1` and
  `LLM_MODEL=meta-llama/llama-3.3-70b-instruct:free`.

## 4. Run it locally

```bash
uvicorn app.main:app --reload --port 8000
```

Test it:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Hiring a mid-level Java developer who also writes SQL"}]
}'
```

## 5. Run the tests

```bash
pytest tests/test_agent.py -v      # offline, no API key needed - tests the pipeline logic
```

Once you've downloaded the assignment's 10 conversation traces and unzipped them into
`tests/traces/`, open ONE of them to see its real field names, adjust `load_trace()` in
`scripts/eval_traces.py` to match, then:

```bash
uvicorn app.main:app --port 8000 &
python scripts/eval_traces.py --api-url http://localhost:8000 --traces-dir tests/traces
```

## 6. Deploy (pick one - all free)

### Render (easiest, `render.yaml` is already set up)
1. Push this folder to a GitHub repo.
2. https://render.com -> New -> Blueprint -> point at your repo (Render reads `render.yaml`).
3. Set the `LLM_API_KEY` env var in the Render dashboard (marked `sync: false` so it asks you).
4. Wait for the build; note the first `/health` call can take up to 2 minutes (cold start) -
   the assignment explicitly allows for this.

### Fly.io / Railway / Hugging Face Spaces (Docker)
All three accept the included `Dockerfile` directly:
```bash
fly launch          # or: railway up   /  create a Docker Space on HF and push
fly secrets set LLM_API_KEY=... LLM_BASE_URL=https://api.groq.com/openai/v1 LLM_MODEL=llama-3.3-70b-versatile
```

## 7. Submit

- Public API endpoint URL of your deployed `/health` and `/chat`.
- `APPROACH.md` (or export it to PDF/Word) - design choices, trade-offs, what didn't work.
- Via the Qualtrics form linked in the assignment email.
