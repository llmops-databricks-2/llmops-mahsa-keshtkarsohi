# AI Daily Digest

An LLMOps project on Databricks: an AI agent that searches the public web every day for articles
about AI, writes a short summary, and sends it to me as a PDF report with a notification.

Keeping up with AI is hard: model releases, research, and product announcements appear every day
across many websites. This agent gathers that information automatically, so I get one readable
report each morning instead of checking dozens of sites.

## What it does

1. **Search**: every morning, a scheduled Databricks job collects new AI articles from the public
   web, including all relevant articles from OpenAI and Anthropic.
2. **Filter**: duplicates and articles that are not about AI are removed.
3. **Summarize**: an LLM agent reads the day's articles, groups them by topic, and writes a
   digest with links to the original articles.
4. **Report**: the digest is turned into a PDF file, such as `ai_digest_2026-10-23.pdf`.
5. **Notify**: I receive a notification with the report every day.
6. **Evaluate and monitor**: the agent is traced and evaluated with MLflow for summary quality,
   faithfulness to the sources, and cost.

## Output

The output is not a table: it is a **daily PDF report**, delivered to me with a notification.

Each report contains:

- the date and the number of articles found
- a short overview of the most important AI news of the day
- sections per topic, such as model releases, research, tools, and business news
- for every article, a two- or three-sentence summary and a link to the original

Past reports are kept in a Unity Catalog volume, so I can look back at earlier days.

## Data sources

All sources are public web content. The focus is on articles from the major AI labs, completed
with research papers and AI news from the wider web.

| Source | What it provides |
|---|---|
| OpenAI (news, research, and engineering articles) | Model releases, research results, product and safety updates |
| Anthropic (news, research, and engineering articles) | Model releases, research results, product and safety updates |
| Other AI lab and company blogs, such as Google DeepMind, Meta AI, Hugging Face, and Databricks | Releases, technical posts, open-source tools |
| arXiv (cs.AI, cs.LG, cs.CL) | New research papers: titles, abstracts, authors |
| Public AI news on the web, such as Hacker News and tech news sites | AI stories and discussions from the wider community |

Only publicly available articles are collected, using official feeds and APIs where they exist.
The report contains short summaries and links, and always points to the original article.

## Data contract

Behind the report, every collected article is stored with the same columns, so the summary step
does not depend on the source and the agent never reports the same article twice.

| Column | Description |
|---|---|
| `source` | Where the article comes from, such as `openai`, `anthropic`, `arxiv`, or `hackernews` |
| `item_id` | Unique id of the article, such as a hash of its URL |
| `title` | Title of the article |
| `url` | Link to the original article |
| `published` | Publication time |
| `text` | Article text, abstract, or description |
| `authors` | Authors, when available |
| `ingested_at` | When the article was collected |

## Architecture

```
Public web (OpenAI, Anthropic, other AI sources)
        │
        ▼
Daily search and collection job ──► Articles in Unity Catalog (no duplicates)
                                                │
                                                ▼
                                Summary agent (course_ops.gateway.chat)
                                                │
                                                ▼
                       PDF report in a Unity Catalog volume ──► Notification to me
                                                │
                                                ▼
                                   MLflow traces and evaluation
```

| Component | Databricks feature |
|---|---|
| Daily scheduling | Serverless jobs, deployed with Declarative Automation Bundles |
| Article storage | Unity Catalog, Delta table in `llmops_dev.mahsa_keshtkarsohi` |
| LLM calls | AI Gateway model services (`course_ops.gateway.chat`, `course_ops.gateway.embed`) |
| PDF reports | Unity Catalog volume |
| Notification | Daily email or message with the report |
| Search over past articles | Embeddings stored in Lakebase with pgvector |
| Tracing and evaluation | MLflow experiments per environment |
| Environments | `dev`, `acc`, and `prd` bundle targets |

## Roadmap

- [x] Project setup: environment, CLI authentication, Databricks Connect
- [ ] Daily collection of AI articles from OpenAI, Anthropic, and other public sources
- [ ] Summary agent that writes the daily digest
- [ ] PDF report generation
- [ ] Daily notification with the report
- [ ] Embeddings and search over past articles (ask questions such as "what did Anthropic release this month?")
- [ ] Evaluation of summary quality with MLflow
- [ ] Deployment to acceptance and production with CI/CD

---

## Course setup

Everything in the course workspace was provisioned before this repo was created, and every id
is already in `project_config.yml`.

| What | Value |
|---|---|
| Schema in development, which I manage | `llmops_dev.mahsa_keshtkarsohi` |
| Schema in acceptance, read-only | `llmops_acc.mahsa_keshtkarsohi` |
| Schema in production, read-only | `llmops_prd.mahsa_keshtkarsohi` |
| SQL warehouse id | `be8f0e8cc38ba3f4` |
| Usage policy id | `20b687b6-7c89-3e81-9ce3-8177e00936fa` |
| MLflow experiments | Dev: `/llmops/mahsa_keshtkarsohi-dev`; acc: `/llmops/mahsa_keshtkarsohi-acc`; prd: `/llmops/mahsa_keshtkarsohi-prd` |
| Chat model service | `course_ops.gateway.chat` (GPT-OSS-120B) |
| Embedding model service | `course_ops.gateway.embed` (GTE Large) |

### Repository structure

- `src/arxiv_curator/`: the Python package; `config.py` loads `project_config.yml`.
- `lecture_1/`, `lecture_2/`, and so on: notebooks copied from the course code.
- `databricks.yml` and `resources/`: the bundle with the `dev`, `acc`, and `prd` targets.
- `pyproject.toml` and `version.txt`: the package and its version.
- `tests/`: tests; `test_config.py` checks that `project_config.yml` loads.
- `.github/workflows/ci.yml`: runs the pre-commit hooks and tests on pull requests and pushes to `main`.

### Getting started

```bash
uv sync --extra dev
uvx pre-commit install
databricks bundle validate -t dev
```