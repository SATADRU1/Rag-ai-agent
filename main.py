import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai 
from dotenv import load_dotenv
import os
import uuid
import datetime
from Data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_type import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult



load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag-ai-agent",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(
        event="rag/ingest_pdf",
    )
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id" , pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)
        


    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(vecs))]
        payloads = [{"text": chunks[i], "source": source_id} for i in range(len(chunks))]
        QdrantStorage().upsert(ids,vecs,payloads)
        return RAGUpsertResult(ingested=len(chunks))







    chunks_and_src = await ctx.step.run("load_pdf",lambda: _load(ctx),output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("upsert_pdf",lambda: _upsert(chunks_and_src),output_type=RAGUpsertResult)
    return ingested.model_dump()


app = FastAPI()


inngest.fast_api.serve(app,inngest_client,functions=[rag_ingest_pdf])