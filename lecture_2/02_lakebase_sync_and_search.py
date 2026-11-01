"""Lecture 2: Lakebase Synced Table & Search Comparison (Semantic, Filtered, Hybrid RRF)."""

import os

from arxiv_curator.config import get_env, load_config
from arxiv_curator.vector_search import VectorSearchManager

if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", "Mahsa")

cfg = load_config("project_config.yml", env=get_env())
vs = VectorSearchManager(cfg)

# Step 3: Create/verify synced table, wait until online, and build pgvector HNSW index
vs.create_or_get_synced_table()
vs.wait_until_online()
vs.build_pgvector_index()

# Step 4: Compare Semantic, Filtered, and Hybrid Search
query = "Zero Data Retention ZDR enterprise safeguards misuse detection"
print(f"\n=== 1. Semantic Search: '{query}' ===")
for r in vs.search_semantic(query, limit=3):
    print(f"  [{r['score']:.3f}] ({r['source']}) {r['title']} [chunk {r['chunk_index']}]")
    print(f"    Context: {r['chunk_context']}")

print(f"\n=== 2. Filtered Vector Search (source='anthropic'): '{query}' ===")
for r in vs.search_semantic(query, limit=3, source="anthropic"):
    print(f"  [{r['score']:.3f}] ({r['source']}) {r['title']} [chunk {r['chunk_index']}]")

hybrid_query = "OLMo-3 RLVR instrumental convergence"
print(f"\n=== 3. Hybrid Search (Vector + FTS with RRF): '{hybrid_query}' ===")
for r in vs.search_hybrid(hybrid_query, limit=3):
    print(
        f"  [RRF={r['rrf_score']:.4f} | vec_rank={r['vec_rank']} | fts_rank={r['fts_rank']}] "
        f"({r['source']}) {r['title']} [chunk {r['chunk_index']}]"
    )
