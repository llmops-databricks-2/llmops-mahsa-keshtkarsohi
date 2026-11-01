# Databricks notebook source
# ruff: noqa: ERA001
# Databricks MAGIC comments are executable notebook cells.
"""Collect the day's AI articles into the ai_articles table of the target's schema."""

# COMMAND ----------

# MAGIC %md
# MAGIC # Collect AI articles
# MAGIC
# MAGIC Runs every morning as the `collect` job. It collects articles published in the last
# MAGIC `lookback_days` days from every source in `arxiv_curator.ingestion` and adds the new ones to
# MAGIC `ai_articles`. Articles already in the table (same `item_id`) are skipped, so a rerun or an
# MAGIC overlapping lookback never creates duplicates.

# COMMAND ----------

from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from databricks.sdk.runtime import dbutils
from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config
from arxiv_curator.ingestion import collect_articles

# The job sets env through its base parameters; by hand the widgets default to dev and 2 days.
dbutils.widgets.text("env", "dev")
dbutils.widgets.text("lookback_days", "2")

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
lookback_days = int(dbutils.widgets.get("lookback_days"))

cfg = load_config("../project_config.yml", env)
table = f"{cfg.full_schema_name}.ai_articles"
logger.info(f"Environment {env}: collecting the last {lookback_days} days into {table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The table and its data contract

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {table} (
        source STRING NOT NULL COMMENT 'Where the article comes from, such as openai, anthropic, arxiv',
        item_id STRING NOT NULL COMMENT 'SHA-256 of the article URL; unique per article',
        title STRING COMMENT 'Title of the article',
        url STRING COMMENT 'Link to the original article',
        published TIMESTAMP COMMENT 'Publication time',
        text STRING COMMENT 'Article text, abstract, or description',
        authors ARRAY<STRING> COMMENT 'Authors, when available',
        ingested_at TIMESTAMP COMMENT 'When the article was collected'
    )
    COMMENT 'AI articles collected daily for the AI Daily Digest'
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Collect and add the new articles

# COMMAND ----------

since = datetime.now(UTC) - timedelta(days=lookback_days)
articles = collect_articles(since)
logger.info(f"Collected {len(articles)} articles published since {since:%Y-%m-%d %H:%M} UTC")

if articles:
    # Same column order and types as the table, so the merge can insert whole rows.
    new_rows = spark.createDataFrame([asdict(article) for article in articles], schema=spark.table(table).schema)
    # One row per item_id: MERGE fails when two source rows match the same key.
    new_rows.dropDuplicates(["item_id"]).createOrReplaceTempView("new_articles")
    merge_result = spark.sql(f"""
        MERGE INTO {table} AS existing
        USING new_articles AS new
        ON existing.item_id = new.item_id
        WHEN NOT MATCHED THEN INSERT *
    """)
    merge_result.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check

# COMMAND ----------

spark.sql(f"""
    SELECT source, count(*) AS articles, max(published) AS newest
    FROM {table}
    GROUP BY source
    ORDER BY articles DESC
""").show(truncate=False)
