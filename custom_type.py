import pydantic

class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str= None
    

class RAGUpsertResult(pydantic.BaseModel):
    ingested: int


class RAGSearchResult(pydantic.BaseModel):
    context: list[str]
    sources: list[str]

class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int   

class ReconcileRecord(pydantic.BaseModel):
    txn_id: str
    utr: str
    settlement_amount: float
    bank_amount: float | None = None
    settled_date: str
    credited_date: str | None = None
    status: str          # matched | amount_mismatch | missing_in_bank | duplicate | delayed
    reason: str

class ReconcileResult(pydantic.BaseModel):
    total_records: int
    matched: int
    match_rate: float
    exceptions: list[ReconcileRecord]
