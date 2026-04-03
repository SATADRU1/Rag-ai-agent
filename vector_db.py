from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class QdrantStorage: 
    def __init__(self, url="http://localhost: 6333", collection="docs", dim=3072):
        self.client =QdrantClient(url=url, timeout=30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=id[i], vector=vector[i], payload=payload[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k=5):
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k,
        )
        context = []
        sources = set()

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            _source = payload.get("source", "")
            if text:
                context.append(text)
                if _source:
                    sources.add(_source)
        
        return {"context": "\n\n".join(context), "sources": list(sources)}


        