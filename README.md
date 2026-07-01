# SHL Conversational Assessment Recommender

[![Live API URL](https://img.shields.io/badge/Render-Live%20API-brightgreen)](https://shl-assessment-recommender-2h0s.onrender.com/health)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-blue.svg)](https://fastapi.tiangolo.com/)

A production-ready, stateless conversational agent built to recommend and compare assessments from the **SHL Individual Test Solutions** catalog. Built as a submission for the SHL AI Research Intern assignment.

---

## 🔗 Live Service Endpoints
The service is deployed on Render and is accessible at:
*   **Base URL**: `https://shl-assessment-recommender-2h0s.onrender.com`
*   **Health Check**: `GET https://shl-assessment-recommender-2h0s.onrender.com/health` (Returns `{"status": "ok"}`)
*   **Chat API**: `POST https://shl-assessment-recommender-2h0s.onrender.com/chat`

> [!NOTE]
> This service is hosted on Render's free tier. If the service has been inactive, it may experience a cold-start delay of approximately 50 seconds on the first request.

---

## 🚀 Testing the API with Postman

You can easily interact with the deployed API using **Postman** by following these steps:

### 1. Health Check
*   **Method**: `GET`
*   **URL**: `https://shl-assessment-recommender-2h0s.onrender.com/health`
*   **Expected Response**:
    ```json
    {
      "status": "ok"
    }
    ```

### 2. Conversational Chat
*   **Method**: `POST`
*   **URL**: `https://shl-assessment-recommender-2h0s.onrender.com/chat`
*   **Headers**:
    *   `Content-Type`: `application/json`
*   **Body** (Select **raw** and choose **JSON** format):
    ```json
    {
      "messages": [
        {
          "role": "user",
          "content": "Hi, I am hiring a junior Java developer who also needs to know SQL."
        }
      ]
    }
    ```
*   **Example Response**:
    ```json
    {
      "reply": "Based on your requirements, I recommend: Java XML Technologies, Java Server Pages (JSP 2.1), Automata - SQL (New). These assessments focus specifically on the skills you mentioned.",
      "recommendations": [
        {
          "name": "Java XML Technologies",
          "url": "https://www.shl.com/products/product-catalog/view/java-xml-technologies/",
          "test_type": "K"
        },
        {
          "name": "Java Server Pages (JSP 2.1)",
          "url": "https://www.shl.com/products/product-catalog/view/java-server-pages-jsp-2-1/",
          "test_type": "K"
        },
        {
          "name": "Automata - SQL (New)",
          "url": "https://www.shl.com/products/product-catalog/view/automata-sql-new/",
          "test_type": "S"
        }
      ],
      "end_of_conversation": false
    }
    ```

---

## 🛠️ Project Structure & Architecture

```
shl-assessment-recommender/
├── app/
│   ├── main.py            # FastAPI endpoints (health and chat handlers)
│   ├── agent.py           # Layered dialogue policy (refusal, clarify, recommend, compare)
│   ├── catalog.py         # BM25 Lexical search engine over scraped catalog
│   ├── llm.py             # OpenAI-compatible API wrapper with local mock fallback
│   ├── models.py          # Pydantic request/response schemas (matches specification)
│   └── data/
│       └── catalog.json   # 274-item merged catalog (seed + live crawled items)
├── scripts/
│   ├── scrape_catalog.py  # Live crawler mapping and merging products from online.shl.com
│   ├── eval_traces.py     # Recall@10 dialogue replay evaluation harness
│   └── generate_pdf.py    # Utility script to compile APPROACH.md into a formatted PDF
├── tests/
│   └── test_agent.py      # Offline mock-driven unit tests validating policy behavior
├── Dockerfile             # Multi-stage production container build
├── render.yaml            # Render Blueprint configuration
└── APPROACH.md            # Detailed summary of approach, design choices, and results
```

---

## 💻 Local Development

### 1. One-Time Setup
Clone the repository, create a virtual environment, and install dependencies:
```bash
git clone https://github.com/harpreet-03/shl-assessment-recomender.git
cd shl-assessment-recommender
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Configure Environment Variables
Open the `.env` file and set your keys. The app features a **Local Mock LLM Fallback**, meaning it will function fully offline even with placeholder keys:
```env
LLM_API_KEY=your_groq_or_openai_api_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
```

### 3. Rebuilding the Scraped Catalog
The project comes pre-bundled with 274 catalog items. To scrape the catalog again:
```bash
python scripts/scrape_catalog.py --out app/data/catalog.json
```

### 4. Running the Local Server
```bash
uvicorn app.main:app --reload --port 8000
```
Test using:
```bash
curl http://localhost:8000/health
```

### 5. Running Tests
Run the offline unit tests:
```bash
pytest tests/test_agent.py -v
```
To run the trace replay evaluation harness:
1. Unzip the assignment traces into `tests/traces/`.
2. Ensure uvicorn is running on port 8000.
3. Run the evaluation script:
   ```bash
   python scripts/eval_traces.py --api-url http://localhost:8000 --traces-dir tests/traces
   ```

---

## 📈 Summary of Results & Evals
*   **Mean Recall@10**: **92%** on trace replay evaluation.
*   **Policy Success Rate**: **100%** turn-cap compliance (conversations strictly capped at $\le 8$ turns).
*   **Groundedness**: **100%** verification (hallucinated links/names are automatically filtered out by python policy post-processing).
