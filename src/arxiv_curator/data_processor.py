"""Chunking and Anthropic Contextual Retrieval processor for AI Daily Digest articles."""

from __future__ import annotations

import json
import re

from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI


def chunk_text_with_overlap(text: str, chunk_size: int = 1000, overlap_pct: float = 0.20) -> list[str]:
    """Split text into fixed-size character chunks with X% overlap, snapping to word boundaries."""
    clean = " ".join((text or "").split())
    if len(clean) <= chunk_size:
        return [clean] if clean else []

    overlap = int(chunk_size * overlap_pct)
    chunks: list[str] = []
    start = 0

    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        if end < len(clean):
            space = clean.rfind(" ", start + chunk_size // 2, end)
            if space != -1:
                end = space
        chunks.append(clean[start:end].strip())
        if end >= len(clean):
            break
        next_start = max(start + 1, end - overlap)
        space_start = clean.find(" ", next_start, end)
        start = space_start + 1 if space_start != -1 else next_start

    return [c for c in chunks if c]


def get_gateway_client(w: WorkspaceClient | None = None) -> OpenAI:
    """Return an OpenAI-compatible client pointed at Databricks Unity AI Gateway."""
    w = w or WorkspaceClient()
    token = w.config.authenticate()["Authorization"].split(" ")[1]
    return OpenAI(
        api_key=token,
        base_url=f"{w.config.host.rstrip('/')}/ai-gateway/mlflow/v1",
    )


def generate_article_chunk_contexts(
    client: OpenAI,
    llm_endpoint: str,
    title: str,
    source: str,
    full_doc: str,
    chunks: list[str],
) -> list[str]:
    """Anthropic Contextual Retrieval: generate short situating contexts for all chunks of a long article in one call."""
    if len(chunks) <= 2:
        return [f"From {source} article '{title}' (part {i + 1} of {len(chunks)})." for i in range(len(chunks))]

    chunks_xml = "\n".join(f"<chunk index='{i}'>\n{c}\n</chunk>" for i, c in enumerate(chunks))
    prompt = (
        f"<document title='{title}' source='{source}'>\n{full_doc}\n</document>\n\n"
        f"Here are the {len(chunks)} chunks of this document:\n{chunks_xml}\n\n"
        f"Return a JSON array of {len(chunks)} strings. Each string must be a 1-sentence "
        "succinct context situating that chunk within the overall document for search retrieval. "
        "Return ONLY the raw JSON array."
    )
    resp = client.chat.completions.create(
        model=llm_endpoint,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
    )
    content = (resp.choices[0].message.content or "").strip()
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, list) and len(parsed) == len(chunks):
            return [str(x) for x in parsed]
    return [f"From {source} article '{title}' (part {i + 1} of {len(chunks)})." for i in range(len(chunks))]


def build_contextual_chunks(
    articles: list[dict],
    client: OpenAI,
    llm_endpoint: str,
    embedding_endpoint: str,
    chunk_size: int = 1000,
    overlap_pct: float = 0.20,
) -> list[dict]:
    """Chunk articles with fixed size + overlap, add Anthropic context, and compute embeddings."""
    records: list[dict] = []
    for art in articles:
        raw_chunks = chunk_text_with_overlap(art["text"], chunk_size, overlap_pct)
        contexts = generate_article_chunk_contexts(
            client, llm_endpoint, art["title"], art["source"], art["text"], raw_chunks
        )
        for idx, (raw_chunk, ctx) in enumerate(zip(raw_chunks, contexts, strict=True)):
            records.append(
                {
                    "chunk_id": f"{art['item_id']}_{idx}",
                    "item_id": art["item_id"],
                    "chunk_index": idx,
                    "source": art["source"],
                    "title": art["title"],
                    "url": art["url"],
                    "published": art["published"],
                    "authors": art["authors"],
                    "raw_chunk_text": raw_chunk,
                    "chunk_context": ctx,
                    "text": f"[Context: {ctx}]\n\n{raw_chunk}",
                }
            )

    embeddings = embed_texts(client, embedding_endpoint, [r["text"] for r in records])
    for rec, emb in zip(records, embeddings, strict=True):
        rec["embedding"] = emb
    return records


def embed_texts(client: OpenAI, embedding_endpoint: str, texts: list[str], batch_size: int = 128) -> list[list[float]]:
    """Compute embeddings in batches via Unity AI Gateway (course_ops.gateway.embed)."""
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=embedding_endpoint, input=batch)
        embeddings.extend([item.embedding for item in resp.data])
    logger.info(f"Embedded {len(embeddings)} chunks via {embedding_endpoint}")
    return embeddings
