# Settlement Reconciliation Agent

**Track:** Razorpay AI Buildathon — AI Finance Controller (Settlement Q&A Agent)

An agent that reconciles Razorpay settlement records against bank statements,
flags exceptions with a reason, and answers natural-language questions about
why a specific transaction is or isn't settled — grounded in both the
reconciliation result and the merchant's settlement policy documents.

## What it actually does

1. **Reconciles** a settlement report against a bank statement, classifying
   every record as `matched`, `amount_mismatch`, `missing_in_bank`,
   `duplicate`, or `delayed` — with a human-readable reason for each.

2. **Answers transaction-specific questions** ("why wasn't txn X settled?")
   by combining the reconciliation result with relevant policy text, using
   retrieval-augmented generation over uploaded policy PDFs.

3. **Reports honest accuracy** — evaluated against a labeled synthetic
   dataset, not cherry-picked examples.

## Results (from `evaluate.py`)

- Graded against 60 synthetic settlement records with known ground truth
- **Accuracy: 100.0%**
- Every misclassification is printed, not hidden — see `evaluate.py` output

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full diagram and data flow.

**Tech stack:**

- **Reconciliation engine** — pure Python, rule-based matching with tolerance
  windows (`reconcile.py`)
- **RAG pipeline** — PyMuPDF text extraction → LlamaIndex chunking →
  MiniLM embeddings → Qdrant vector search → Groq LLM generation
- **Orchestration** — Inngest (step functions, retries, observability)
- **API** — FastAPI
- **UI** — Streamlit (3 tabs: Dashboard, Transaction Q&A, Policy Chat)

## Running it locally

1. `uv sync` (or `pip install -r requirements.txt`)
2. Set `GROQ_API_KEY` in `.env`
3. Generate synthetic data: `python data/generate_settlement_data.py`
4. Run reconciliation: `python reconcile.py`
5. Grade accuracy: `python evaluate.py`
6. Start Qdrant: `docker run -d --name qdrantRagDb -p 6333:6333 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant`
7. Start the API: `uvicorn main:app --reload`
8. Start Inngest dev server: `npx inngest-cli@latest dev -u http://localhost:8000/api/inngest --no-discovery`
9. Start the UI: `streamlit run streamlit_app.py`

## Live demo

https://rag-ai-agent-nywmctur8xwxv2y2btzude.streamlit.app/

## Why this matters

Manual settlement reconciliation is slow and error-prone at scale. This agent
gives a merchant (or their finance team) an instant, auditable answer to
"where's my money and why" — with every decision traceable to a reason,
not a black box.
