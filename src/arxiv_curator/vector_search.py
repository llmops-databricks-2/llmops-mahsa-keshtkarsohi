"""Minimal Lakebase Postgres vector store manager for AI Daily Digest chunks."""

from __future__ import annotations

import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import (
    SyncedTable,
    SyncedTableState,
    SyncedTableSyncedTableSpec,
    SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy,
)
from loguru import logger

from arxiv_curator.config import ProjectConfig
from arxiv_curator.data_processor import embed_texts, get_gateway_client


class VectorSearchManager:
    """Manages Lakebase synced table, pgvector index, and Semantic/Filtered/Hybrid search."""

    def __init__(
        self,
        config: ProjectConfig,
        source_table: str = "ai_article_chunks",
        embedding_dim: int = 1024,
    ) -> None:
        self.config = config
        self.w = WorkspaceClient()
        self.project_name = config.lakebase_instance
        self.source_table = f"{config.full_schema_name}.{source_table}"
        self.synced_table = f"{self.source_table}_synced"
        self.pg_table = f'"{config.schema}"."{source_table}_synced"'
        self.embedding_dim = embedding_dim
        self._conn: psycopg.Connection | None = None

    def connect(self) -> psycopg.Connection:
        """Connect to Lakebase Postgres using a generated OAuth token."""
        if self._conn is None or self._conn.closed:
            parent = f"projects/{self.project_name}/branches/production"
            ep = next(iter(self.w.postgres.list_endpoints(parent=parent)))
            host = ep.status.hosts.host
            cred = self.w.postgres.generate_database_credential(endpoint=ep.name)
            user = self.w.current_user.me().user_name
            self._conn = psycopg.connect(
                host=host,
                port=5432,
                dbname="databricks_postgres",
                user=user,
                password=cred.token,
                sslmode="require",
                autocommit=True,
            )
        return self._conn

    def create_or_get_synced_table(self) -> SyncedTable:
        """Create or retrieve the Lakebase Autoscaling synced table from Unity Catalog."""
        name = f"synced_tables/{self.synced_table}"
        try:
            tbl = self.w.postgres.get_synced_table(name=name)
            logger.info(f"Synced table exists: {self.synced_table}")
            return tbl
        except Exception:
            logger.info(f"Creating synced table: {self.synced_table}")
            op = self.w.postgres.create_synced_table(
                synced_table_id=self.synced_table,
                synced_table=SyncedTable(
                    spec=SyncedTableSyncedTableSpec(
                        source_table_full_name=self.source_table,
                        primary_key_columns=["chunk_id"],
                        scheduling_policy=SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy.SNAPSHOT,
                        branch=f"projects/{self.project_name}/branches/production",
                        postgres_database="databricks_postgres",
                        create_database_objects_if_missing=True,
                    )
                ),
            )
            return op.wait()

    def wait_until_online(self, poll_seconds: int = 10, timeout_seconds: int = 600) -> SyncedTableState:
        """Poll until the Lakebase synced table finishes syncing."""
        import time

        name = f"synced_tables/{self.synced_table}"
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            tbl = self.w.postgres.get_synced_table(name=name)
            state = tbl.status.detailed_state if tbl.status else None
            logger.info(f"Sync state: {state}")
            if state in {
                SyncedTableState.SYNCED_TABLE_ONLINE,
                SyncedTableState.SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE,
                SyncedTableState.SYNCED_TABLE_ONLINE_CONTINUOUS_UPDATE,
            }:
                return state
            if "FAILED" in str(state) or "OFFLINE" in str(state):
                raise RuntimeError(f"Sync failed: {tbl.status.message if tbl.status else state}")
            time.sleep(poll_seconds)
        raise TimeoutError(f"Timed out waiting for {self.synced_table}")

    def build_pgvector_index(self) -> None:
        """Enable pgvector and create an HNSW cosine index on the synced table."""
        conn = self.connect()
        dim = self.embedding_dim
        idx_name = f"{self.config.schema}_ai_chunks_emb_hnsw_idx"
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {idx_name} ON {self.pg_table} "
            f"USING hnsw (((embedding::text)::vector({dim})) vector_cosine_ops)"
        )
        logger.info(f"Created pgvector HNSW index: {idx_name}")

    def embed_query(self, query: str) -> str:
        """Embed a search query via course_ops.gateway.embed and format as pgvector literal."""
        client = get_gateway_client(self.w)
        vec = embed_texts(client, self.config.embedding_endpoint, [query])[0]
        return "[" + ",".join(f"{v:.7f}" for v in vec) + "]"

    def search_semantic(
        self, query: str, limit: int = 5, source: str | None = None, year: int | None = None
    ) -> list[dict]:
        """Run semantic or metadata-filtered vector search on Lakebase pgvector."""
        qvec = self.embed_query(query)
        dim = self.embedding_dim
        vec_expr = f"((embedding::text)::vector({dim}))"
        clauses, params = ["embedding IS NOT NULL"], []
        if source:
            clauses.append("source = %s")
            params.append(source)
        if year:
            clauses.append("year = %s")
            params.append(year)
        where_sql = " AND ".join(clauses)
        sql = f"""
            SELECT chunk_id, source, title, chunk_index, chunk_context, raw_chunk_text,
                   1 - ({vec_expr} <=> %s::vector({dim})) AS score
            FROM {self.pg_table}
            WHERE {where_sql}
            ORDER BY {vec_expr} <=> %s::vector({dim})
            LIMIT %s
        """
        rows = self.connect().execute(sql, [qvec, *params, qvec, limit]).fetchall()
        return [
            {
                "chunk_id": r[0],
                "source": r[1],
                "title": r[2],
                "chunk_index": r[3],
                "chunk_context": r[4],
                "raw_chunk_text": r[5],
                "score": float(r[6]),
            }
            for r in rows
        ]

    def search_hybrid(self, query: str, limit: int = 5, rrf_k: int = 60) -> list[dict]:
        """Run Hybrid Search combining pgvector cosine similarity + Postgres FTS with RRF."""
        qvec = self.embed_query(query)
        dim = self.embedding_dim
        vec_expr = f"((embedding::text)::vector({dim}))"
        pool = limit * 5
        sql = f"""
            WITH vec AS (
                SELECT chunk_id,
                       ROW_NUMBER() OVER (ORDER BY {vec_expr} <=> %s::vector({dim})) AS r_vec
                FROM {self.pg_table}
                WHERE embedding IS NOT NULL
                LIMIT %s
            ),
            fts AS (
                SELECT chunk_id,
                       ROW_NUMBER() OVER (
                           ORDER BY ts_rank_cd(to_tsvector('english', text), websearch_to_tsquery('english', %s)) DESC
                       ) AS r_fts
                FROM {self.pg_table}
                WHERE to_tsvector('english', text) @@ websearch_to_tsquery('english', %s)
                LIMIT %s
            ),
            rrf AS (
                SELECT COALESCE(vec.chunk_id, fts.chunk_id) AS chunk_id,
                       COALESCE(1.0 / (%s + vec.r_vec), 0.0) + COALESCE(1.0 / (%s + fts.r_fts), 0.0) AS rrf_score,
                       vec.r_vec,
                       fts.r_fts
                FROM vec FULL OUTER JOIN fts ON vec.chunk_id = fts.chunk_id
            )
            SELECT c.chunk_id, c.source, c.title, c.chunk_index, c.chunk_context, c.raw_chunk_text,
                   rrf.rrf_score, rrf.r_vec, rrf.r_fts
            FROM rrf
            JOIN {self.pg_table} c ON c.chunk_id = rrf.chunk_id
            ORDER BY rrf.rrf_score DESC
            LIMIT %s
        """
        rows = self.connect().execute(sql, [qvec, pool, query, query, pool, rrf_k, rrf_k, limit]).fetchall()
        return [
            {
                "chunk_id": r[0],
                "source": r[1],
                "title": r[2],
                "chunk_index": r[3],
                "chunk_context": r[4],
                "raw_chunk_text": r[5],
                "rrf_score": float(r[6]),
                "vec_rank": r[7],
                "fts_rank": r[8],
            }
            for r in rows
        ]
