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
    def _load(ctx: inngest.Context) -> dict:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id" , pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id).model_dump()
        


    def _upsert(chunks_and_src: dict) -> dict:
        chunks = chunks_and_src["chunks"]
        source_id = chunks_and_src["source_id"]
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(vecs))]
        payloads = [{"text": chunks[i], "source": source_id} for i in range(len(chunks))]
        QdrantStorage().upsert(ids,vecs,payloads)
        return RAGUpsertResult(ingested=len(chunks)).model_dump()







    chunks_and_src = await ctx.step.run("load_pdf", lambda: _load(ctx))
    ingested = await ctx.step.run("upsert_pdf", lambda: _upsert(chunks_and_src))
    return ingested



@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question:str,top_k:int=5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec,top_k)
        return RAGSearchResult(context=found["context"],sources=found["sources"]).model_dump()
    

    question = ctx.event.data["question"]
    top_k = ctx.event.data.get("top_k",5)
    found = await ctx.step.run("embed_and_search", lambda: _search(question,top_k), output_type=RAGSearchResult)
    context_block = "\n\n".join(f"- {c}" for c in found.context)
    user_context = (
        "use the following context to answer the question \n\n"
        f"{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    adapter = ai.OpenAI.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),      
        model="gpt-4o-mini"
    )
    


    res = await ctx.step.ai.infer(
        "llm answer the question based on the context",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages":[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_context},
            ]
        }
    ) 


    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer":answer,"sources":found.sources, "num_context":len(found.context)}

app = FastAPI()


inngest.fast_api.serve(app,inngest_client,functions=[rag_ingest_pdf,rag_query_pdf_ai])


