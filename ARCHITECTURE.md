# Architecture

```mermaid
flowchart TD
    A[Settlement Report CSV] --> C[reconcile.py]
    B[Bank Statement CSV] --> C
    C --> D{Match?}
    D -->|Yes| E[matched]
    D -->|No| F[Exception: amount_mismatch /<br/>missing_in_bank / duplicate / delayed]

    G[Policy PDFs] --> H[PyMuPDF extract]
    H --> I[LlamaIndex chunking]
    I --> J[MiniLM embeddings]
    J --> K[(Qdrant vector DB)]

    L[User question: 'Why wasn't txn X settled?'] --> M[main.py: rag_query_txn_status]
    F --> M
    K --> M
    M --> N[Groq compound-mini]
    N --> O[Answer + cited policy sources]

    P[Streamlit Dashboard] --> C
    Q[Streamlit Transaction Q&A] --> M
    R[Streamlit Policy Chat] --> K
```

## Data flow summary

1. **Reconciliation path:** two CSVs → rule-based matcher → classified
   records with reasons → dashboard + exception list

2. **Knowledge path:** policy PDFs → chunked → embedded → stored in Qdrant
   for semantic search

3. **Answer path:** a question about a transaction pulls both the
   reconciliation reason and relevant policy text, then an LLM combines
   them into one grounded answer

4. **Orchestration:** every multi-step operation (ingest, query, txn lookup)
   runs as an Inngest function — retryable, observable, and step-isolated
