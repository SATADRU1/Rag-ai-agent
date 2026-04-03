import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai 
from dotenv import load_dotenv
import os
import uuid
import datetime


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
    return {"Hello": "World"}



app = FastAPI()


inngest.fast_api.serve(app,inngest_client,functions=[rag_ingest_pdf])