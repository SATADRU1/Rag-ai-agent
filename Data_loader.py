from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF — much better text extraction
from llama_index.core.node_parser import SentenceSplitter

# Free local embedding model — no API key needed
# all-MiniLM-L6-v2 produces 384-dimensional vectors
_model = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = 384

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str):
    doc = fitz.open(path)
    texts = []
    for page in doc:
        text = page.get_text("text")  # extracts with proper spacing
        if text and text.strip():
            texts.append(text.strip())
    doc.close()
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = _model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()