# 💰 Settlement Reconciliation Agent

> **Razorpay AI Buildathon — AI Finance Controller Track**

An AI agent that reads your Razorpay settlement report and bank statement,
finds every mismatch, and answers the question **"where is my money and why?"**
in plain English — backed by your own policy documents.

---

## 🧒 Explain It Like I'm 5

Imagine Razorpay says it sent you ₹500.
But your bank account only got ₹480.
Where did ₹20 go? Why?

This agent:
1. **Reads both statements** — what Razorpay claims vs what the bank received
2. **Finds every difference** — wrong amounts, missing payments, duplicates, late arrivals
3. **Explains each one** in plain English — using your own policy documents as the source of truth

---

## ✅ What It Actually Does

| Feature | Description |
|---|---|
| 📊 **Reconciliation Dashboard** | Compare settlement vs bank statement in one click |
| 💳 **Transaction Q&A** | Ask why any specific transaction is flagged |
| 📄 **Policy PDF Chat** | Upload any PDF and ask questions about it |
| 🎯 **Honest Accuracy** | Evaluated against 60 labeled records — **100% accuracy** |

---

## 🏗️ Architecture

```
Settlement Report CSV ──┐
                        ├──▶ reconcile.py ──▶ matched / exception + reason
Bank Statement CSV ─────┘

Policy PDFs ──▶ PyMuPDF ──▶ Chunks ──▶ MiniLM Embeddings ──▶ Qdrant DB
                                                                    │
User Question ──▶ Embed ──▶ Search Qdrant ──▶ Top chunks ──▶ Groq LLM
                                                                    │
Transaction ID ──▶ reconcile.py ──▶ Exception reason ──────────────┘
                                                                    │
                                                            Final Answer ✅
```

> 📐 For the full detailed Mermaid flowchart see [ARCHITECTURE.md](./ARCHITECTURE.md)

**Tech Stack:**
- 🐍 **Python** — reconciliation engine, data generation, evaluation
- ⚡ **FastAPI** — backend API server
- 🔁 **Inngest** — step-by-step job orchestration with retries
- 🗄️ **Qdrant** — vector database (stores PDF knowledge)
- 🤖 **Groq** — LLM for generating answers (`groq/compound-mini`)
- 🧠 **MiniLM** — local embedding model (no API cost)
- 🎨 **Streamlit** — 3-tab web dashboard

---

## 📁 Project Structure

```
Rag-ai-agent/
├── main.py                          ← FastAPI + 3 Inngest functions
├── reconcile.py                     ← matching engine (the brain)
├── evaluate.py                      ← accuracy grader
├── streamlit_app.py                 ← 3-tab web UI
├── Data_loader.py                   ← PDF text extraction + chunking
├── vector_db.py                     ← Qdrant read/write
├── custom_type.py                   ← Pydantic data models
├── data/
│   ├── generate_settlement_data.py  ← generates CSVs + ground truth
│   ├── generate_policy_docs.py      ← generates policy PDFs
│   ├── settlement_report.csv        ← what Razorpay claims it settled
│   ├── bank_statement.csv           ← what actually landed in the bank
│   └── ground_truth.csv             ← true labels for grading
├── uploads/                         ← policy PDFs go here
└── qdrant_storage/                  ← Qdrant's persistent data
```

---

## 🚀 How to Run Locally (Step by Step)

### Prerequisites
Make sure these are installed on your machine:
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Node.js](https://nodejs.org/) (for Inngest CLI)
- [uv](https://docs.astral.sh/uv/) — Python package manager

---

### Step 1 — Install dependencies

```powershell
cd "e:\Project\Production level projects\production ready rag ai agent\Rag-ai-agent"
uv sync
```

---

### Step 2 — Set your API key

Create a `.env` file in `Rag-ai-agent/` with:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com)

---

### Step 3 — Start Qdrant (vector database)

```powershell
docker run -d --name qdrantRagDb -p 6333:6333 `
  -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

> If you see "container name already in use", it's already running. Skip this step.

---

### Step 4 — Generate synthetic data

```powershell
uv run python data/generate_settlement_data.py
uv run python data/generate_policy_docs.py
```

This creates:
- `data/settlement_report.csv` — 60 fake Razorpay transactions
- `data/bank_statement.csv` — bank records with intentional mismatches
- `data/ground_truth.csv` — true labels for accuracy testing
- `uploads/settlement_timeline_policy.pdf`
- `uploads/amount_mismatch_policy.pdf`
- `uploads/duplicate_and_missing_policy.pdf`

---

### Step 5 — Open 3 terminals and run these

**Terminal 1 — FastAPI backend:**
```powershell
cd "e:\Project\Production level projects\production ready rag ai agent\Rag-ai-agent"
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — Inngest dev server:**
```powershell
cd "e:\Project\Production level projects\production ready rag ai agent\Rag-ai-agent"
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

**Terminal 3 — Streamlit UI:**
```powershell
cd "e:\Project\Production level projects\production ready rag ai agent\Rag-ai-agent"
uv run streamlit run streamlit_app.py
```

Then open **http://localhost:8501** in your browser.

> ⚠️ Always use `uv run` for Python commands. Never use bare `python` or `streamlit` — those use system Python which doesn't have the project's packages.

---

## 🎮 How to Use the App (Test Flow)

Follow this order for the best demo experience:

### 1️⃣ PDF Policy Chat tab — load the knowledge
- Click **"Browse files"**
- Upload `settlement_timeline_policy.pdf` from the `uploads/` folder
- Wait for the green ✅ **"Triggered ingestion for: ..."** message
- Repeat for `amount_mismatch_policy.pdf` and `duplicate_and_missing_policy.pdf`

### 2️⃣ Dashboard tab — run reconciliation
- Click **"Run Reconciliation"**
- See the summary: Total Records / Matched / Match Rate
- Scroll the exceptions table — look at the Status and Reason columns
- **Copy a `txn_id`** from any row (e.g. `txn_e97beebd14`)

### 3️⃣ Ask About a Transaction tab — get AI explanation
- Paste the `txn_id` you copied
- Type a question like: `Why is this transaction flagged?`
- Click **Ask**
- Get a full answer with the exact amounts + what the policy says about it

---

## 🗄️ Qdrant — Managing Your Stored PDFs

### See which PDFs are currently stored

```powershell
cd "e:\Project\Production level projects\production ready rag ai agent\Rag-ai-agent"
uv run python -c "
from qdrant_client import QdrantClient
client = QdrantClient(host='localhost', port=6333)
result = client.scroll('docs', limit=200, with_payload=True)
sources = set(p.payload['source'] for p in result[0])
print(f'Total chunks stored: {len(result[0])}')
print('PDFs currently in Qdrant:')
for s in sources:
    print(f'  - {s}')
"
```

### Delete a specific PDF from Qdrant

Replace `YOUR_PDF_NAME.pdf` with the exact name shown above:

```powershell
uv run python -c "
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
client = QdrantClient(host='localhost', port=6333)
client.delete(
    collection_name='docs',
    points_selector=Filter(
        must=[FieldCondition(key='source', match=MatchValue(value='YOUR_PDF_NAME.pdf'))]
    )
)
print('Deleted successfully')
"
```

### Delete ALL data and start fresh

```powershell
uv run python -c "
from qdrant_client import QdrantClient
client = QdrantClient(host='localhost', port=6333)
client.delete_collection('docs')
print('All data cleared from Qdrant')
"
```

> After clearing all data, re-upload your PDFs through the PDF Policy Chat tab.

---

## 📊 Accuracy Results

```powershell
uv run python evaluate.py
```

```
Total records graded : 60
Correctly classified : 60
Accuracy             : 100.0%
Misclassified        : 0
```

Evaluated against 60 synthetic records with known ground truth labels.
Every misclassification is printed — nothing is hidden.

---

## 🔍 Exception Types Explained

| Status | What it means |
|---|---|
| `matched` | Settlement and bank records agree ✅ |
| `amount_mismatch` | Razorpay says ₹X but bank received ₹Y |
| `missing_in_bank` | Razorpay says settled but bank has no record |
| `duplicate` | Bank credited the same UTR more than once |
| `delayed` | Money arrived more than 2 days after settlement date |

---

## 🌐 Live Demo

https://rag-ai-agent-nywmctur8xwxv2y2btzude.streamlit.app/

---

## 💡 Why This Matters

Manual settlement reconciliation is slow, error-prone, and gives no explanation.
A finance team member has to open two spreadsheets, match rows by hand, and
guess why something doesn't add up.

This agent does it instantly, explains every exception in plain English,
and cites the exact policy that applies — so anyone on the team can
understand the answer without being a finance expert.
