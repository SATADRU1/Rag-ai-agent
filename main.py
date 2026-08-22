import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import os
import uuid
from groq import Groq
from Data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_type import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult
from reconcile import reconcile

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag-ai-agent",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
)


# ---------------------------------------------------------------------------
# Shared helpers — reused by both RAG query and transaction status functions
# ---------------------------------------------------------------------------

def search_context(question: str, top_k: int = 5) -> dict:
    try:
        query_vec = embed_texts([question])[0]
    except Exception as e:
        raise inngest.NonRetriableError(f"Embedding Error: {e}")
    store = QdrantStorage(dim=384)
    found = store.search(query_vec, top_k)
    return RAGSearchResult(context=found["context"], sources=found["sources"]).model_dump()


def llm_answer(context: list, question: str) -> str:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    context_block = "\n\n".join(f"- {c}" for c in context if c)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    completion = groq_client.chat.completions.create(
        model="groq/compound-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer only from the provided context."},
            {"role": "user", "content": user_content},
        ],
        max_tokens=1024,
        temperature=0.2,
    )
    return completion.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Inngest function 1 — Ingest a PDF into the vector DB
# ---------------------------------------------------------------------------

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> dict:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id).model_dump()

    def _upsert(chunks_and_src: dict) -> dict:
        chunks = chunks_and_src["chunks"]
        source_id = chunks_and_src["source_id"]
        try:
            vecs = embed_texts(chunks)
        except Exception as e:
            raise inngest.NonRetriableError(f"Embedding Error: {e}")
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(vecs))]
        payloads = [{"text": chunks[i], "source": source_id} for i in range(len(chunks))]
        QdrantStorage(dim=384).upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks)).model_dump()

    chunks_and_src = await ctx.step.run("load_pdf", lambda: _load(ctx))
    ingested = await ctx.step.run("upsert_pdf", lambda: _upsert(chunks_and_src))
    return ingested


# ---------------------------------------------------------------------------
# Inngest function 2 — Answer a general question about ingested PDFs
# ---------------------------------------------------------------------------

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    question = ctx.event.data["question"]
    top_k = ctx.event.data.get("top_k", 5)

    found = await ctx.step.run("embed_and_search", lambda: search_context(question, top_k))
    answer = await ctx.step.run("llm_answer", lambda: llm_answer(found["context"], question))

    return {"answer": answer, "sources": found["sources"], "num_context": len(found["context"])}


# ---------------------------------------------------------------------------
# Inngest function 3 — Look up a specific transaction and explain its status
# ---------------------------------------------------------------------------

@inngest_client.create_function(
    fn_id="RAG: Query Transaction Status",
    trigger=inngest.TriggerEvent(event="rag/query_txn_status")
)
async def rag_query_txn_status(ctx: inngest.Context):
    def _lookup_txn(txn_id: str) -> dict:
        result = reconcile()
        for record in result.exceptions:
            if record.txn_id == txn_id:
                return record.model_dump()
        return {"txn_id": txn_id, "status": "matched", "reason": "Settlement and bank records align."}

    def _combined_answer(txn_status: dict, policy_context: dict) -> str:
        combined_context = [
            f"Transaction {txn_status['txn_id']} status: {txn_status['status']}. Reason: {txn_status['reason']}"
        ] + policy_context["context"]
        return llm_answer(combined_context, question)

    txn_id = ctx.event.data["txn_id"]
    question = ctx.event.data.get("question", f"Why is transaction {txn_id} showing this status?")

    txn_status = await ctx.step.run("lookup_txn", lambda: _lookup_txn(txn_id))
    policy_context = await ctx.step.run("embed_and_search_policy", lambda: search_context(question))
    answer = await ctx.step.run("llm_answer_txn", lambda: _combined_answer(txn_status, policy_context))

    return {
        "txn_id": txn_id,
        "status": txn_status["status"],
        "answer": answer,
        "sources": policy_context["sources"],
    }


# ---------------------------------------------------------------------------
# FastAPI app — mounts all three Inngest functions
# ---------------------------------------------------------------------------

app = FastAPI()

inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[rag_ingest_pdf, rag_query_pdf_ai, rag_query_txn_status],
)
