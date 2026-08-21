# doc-qa-agent

An end-to-end document question answering project for scientific and technical PDFs / Word documents. The pipeline parses source documents, splits them into chunks, builds dense and lexical retrieval indexes, reranks candidates, and generates grounded answers with source references.

## Background

This project was built to answer questions from two local documents:

- `data/raw/HuMengqing.pdf`
- `data/raw/Report.docx`

The goal is not just to produce answers, but to make the whole RAG pipeline measurable:

- parse documents into structured sections
- chunk sections into indexable units
- retrieve evidence with dense search and BM25
- fuse and rerank candidates
- generate grounded answers
- evaluate retrieval and no-answer behavior
- evaluate retrieved context and answer quality with RAGAS

## Architecture

```mermaid
flowchart TD
    A[PDF / Word documents] --> B[Document parsers]
    B --> C[Sections]
    C --> D[Chunker]
    D --> E[Chunks]
    E --> F[Embedder]
    E --> G[BM25 index]
    F --> H[ChromaDB]
    G --> I[BM25 retriever]
    H --> J[Dense retriever]
    J --> K[Hybrid RRF fusion]
    I --> K
    K --> L[Cross-Encoder reranker]
    L --> M[Prompt builder]
    M --> N[LLM]
    N --> O[Grounded answer + sources]
```

Evaluation flow:

```mermaid
flowchart TD
    A[evaluation/test_queries.json] --> B[Answerable queries]
    A --> C[Unanswerable queries]
    B --> D["Recall@1/3/5/10 + MRR"]
    C --> E[Rejection accuracy]
    D --> F[comparison.json]
    E --> F
```

## Tech Stack

- Python 3.11+
- MinerU API for PDF parsing, table extraction, and formula-to-LaTeX conversion
- `python-docx` for Word parsing
- `langchain-text-splitters` for chunking
- ScaDS.AI-hosted `Qwen/Qwen3-Embedding-4B` for embedding (local `sentence-transformers` with `BAAI/bge-large-en-v1.5` also supported)
- `chromadb` for persistent vector storage
- `rank_bm25` for lexical retrieval
- ScaDS.AI-hosted `Qwen/Qwen3-Reranker-4B` for reranking
- OpenAI-compatible chat API via ScaDS.AI for generation

## Project Layout

```text
config/          YAML configuration
data/raw/        Source PDF and Word files
evaluation/      Annotated test queries and evaluation scripts
prompts/         Prompt templates
scripts/         Utility scripts for annotation
src/core/        Config and logging
src/document/    Parsing and chunking
src/retrieval/   Dense, BM25, hybrid, and reranking
src/generation/  Prompting, LLM wrapper, and RAG pipeline
main.py          CLI entry point
```

## Setup

Create a virtual environment and install the runtime dependencies:

```bash
python -m venv .venv
.venv/bin/pip install python-docx langchain-text-splitters sentence-transformers chromadb rank_bm25 openai python-dotenv pyyaml
```

Create a `.env` file with at least:

```bash
SCADS_API_KEY=your_scads_api_key
HF_TOKEN=your_hugging_face_token
MINERU_API_TOKEN=your_mineru_api_token
```

`HF_TOKEN` is recommended for faster and more reliable model downloads.

## PDF Parsing With MinerU

PDF ingestion uses the official MinerU cloud API. Set `MINERU_API_TOKEN` in
`.env`; the parser requests a pre-signed upload URL, uploads the PDF, submits
an asynchronous extraction task, polls for completion, and downloads the
resulting Markdown archive. The token is read at runtime and is never stored
in `config/config.yaml`.

MinerU receives the PDF content. Review its service terms and your project's
data-handling requirements before parsing sensitive documents. Configure the
cloud endpoint, model version, OCR option, and timeouts under
`document_parsing.mineru` in `config/config.yaml`.

MinerU output preserves formulas as LaTeX and tables as Markdown or HTML; the
parser converts it into the existing ordered text/table section format. There
is no `pdfplumber` fallback: a MinerU request failure stops PDF ingestion with
a clear error.

Word OMML equations are independently converted to inline LaTeX while parsing,
so expressions such as `x_i` and fractions remain searchable.

## Embedding Provider

`embedding.provider` selects between two embedding backends, mirroring the
`local`/`scadsai` switch used for reranking:

- `scadsai` (default): calls ScaDS.AI's OpenAI-compatible `/embeddings`
  endpoint (`embedding.base_url`, default `https://llm.scads.ai/v1`) with
  `embedding.model_name` (default `Qwen/Qwen3-Embedding-4B`) through the same
  `openai` client used by `LLM`. Vectors are L2-normalized locally after the
  request so distances stay comparable to the local provider's output.
- `local`: loads a `sentence-transformers` model (`embedding.model_name`,
  default `BAAI/bge-large-en-v1.5`) and encodes with `normalize_embeddings=True`.

Switching `embedding.provider` or `embedding.model_name` changes the vector
dimension, so it also changes what a Chroma collection can hold. Pick a new
`vector_store.collection_name` whenever you switch embedding models — reusing
the old name upserts incompatible-dimension vectors into an existing
collection and fails. The current default collection is
`doc_chunks_qwen3_embed`; the earlier `doc_chunks` collection (built with
`BAAI/bge-large-en-v1.5`) is left untouched on disk, so reverting
`embedding.provider` to `local` and `vector_store.collection_name` to
`doc_chunks` restores the previous index without re-embedding.

## Reranking Provider

The configured ScaDS.AI reranker posts all fused candidates to
`https://llm.scads.ai/v1/rerank` and expects a `results` list containing one
`index` and `relevance_score` for every supplied candidate. If ScaDS.AI uses a
different path for your account, update `reranking.endpoint` in
`config/config.yaml`.

The client waits at least `reranking.min_request_interval_seconds` between
requests and retries rate-limited (`429`) requests up to
`reranking.max_retries` times. It honors a numeric `Retry-After` response
header when ScaDS.AI provides one.

To return to the local MiniLM reranker, set `reranking.provider` to `local` and
set `reranking.model` to `cross-encoder/ms-marco-MiniLM-L-6-v2`.

## Run

Answer one question from the local documents:

```bash
.venv/bin/python main.py "What classification accuracy did the ResNet26-V2 model achieve?"
```

Annotate evaluation queries:

```bash
.venv/bin/python -m scripts.annotate_chunks
```

By default this reads `evaluation/test_queries_draft.json` and writes
`evaluation/test_queries.json`. Every draft file re-parses and re-chunks the
documents configured under `documents.paths` in `config/config.yaml`, so
whenever the parsing or chunking logic changes (chunk size, section
filtering, etc.), any previously annotated `relevant_chunk_ids` values refer
to stale chunk IDs and the affected query set must be re-annotated against
the new chunks. For a versioned query set (e.g. a fresh draft created after a
chunking change), pass explicit `--input`/`--output` paths under
`evaluation/test_queries/<version>/` instead of relying on the defaults:

```bash
.venv/bin/python -m scripts.annotate_chunks \
  --input evaluation/test_queries/v4/test_queries_draft.json \
  --output evaluation/test_queries/v4/test_queries.json \
  --rerank-candidates
```

This mode is interactive: for each query it prints the query, the reference
answer, and the top `--candidate-top-k` (default 10) hybrid-retrieved
candidates (add `--rerank-candidates` to rerank them with the Cross-Encoder
first), then prompts for the relevant chunk ID(s) or candidate rank number(s)
(comma-separated). Press Enter to keep any existing `relevant_chunk_ids`,
`s` to skip a query, or `q` to quit — progress is written to `--output` after
every query, so an interrupted run resumes where it left off (re-running the
same command skips nothing but lets you leave existing answers unless you
overwrite them).

For a query set too large to annotate one prompt at a time, export the full
candidate list per query to JSON instead, without any terminal prompts:

```bash
.venv/bin/python -m scripts.annotate_chunks \
  --input evaluation/test_queries/v4/test_queries_draft.json \
  --output evaluation/test_queries/v4/test_queries_candidates.json \
  --export-candidates --rerank-candidates
```

Each query record gains a `retrieval_candidates` field (chunk ID + full text
for every retrieved candidate). Open the file, read each query's candidates,
and fill in `relevant_chunk_ids` by hand (or with LLM assistance) using the
chunk IDs shown. Increase `--candidate-top-k` if you suspect the correct
chunk falls outside the default top 10.

After annotation, remove the candidate text before saving the final
evaluation set:

```bash
.venv/bin/python -m scripts.clean_retrieval_candidates \
  --input evaluation/test_queries/v4/test_queries_candidates.json \
  --output evaluation/test_queries/v4/test_queries.json
```

Use `--in-place` instead of `--output <path>` to overwrite the candidates
file directly once you're done reviewing it.

Run retrieval and rejection evaluation with the V4 labels:

```bash
.venv/bin/python -m evaluation.evaluate \
  --input evaluation/test_queries/v4/test_queries.json \
  --results-dir evaluation/results/v4
```

This command writes the following files to `evaluation/results/v4/`:

- `v1_dense.json`
- `v2a_hybrid.json`
- `v2_hybrid_rerank.json`
- `unanswerable_rejection.json`
- `comparison.json`

Run RAGAS evaluation for the same answerable queries:

```bash
.venv/bin/python -m evaluation.evaluate_ragas \
  --input evaluation/test_queries/v4/test_queries.json \
  --results-dir evaluation/results/v4
```

RAGAS uses an LLM as an evaluator through the configured OpenAI-compatible
ScaDS.AI endpoint. Start with a small sample when checking evaluator cost and
compatibility:

```bash
.venv/bin/python -m evaluation.evaluate_ragas \
  --input evaluation/test_queries/v4/test_queries.json \
  --results-dir evaluation/results/v4 \
  --limit 5
```

The RAGAS command writes `evaluation/results/v4/ragas_hybrid_rerank.json` with
per-query and aggregate Context Precision, Context Recall, Faithfulness, and
Factual Correctness scores. It evaluates only answerable queries; the existing
rejection evaluation remains responsible for unanswerable queries.

## Evaluation Data

The current annotated test set contains 50 queries:

- 45 answerable queries
- 5 unanswerable queries

## Evaluation Results

Both evaluations use 45 answerable queries. The 2,000-character baseline and
the 1,000-character configuration use labels generated for their respective
chunking schemes.

### Current V4 Results

Results are stored in `evaluation/results/v4/comparison.json`, using the labels
in `evaluation/test_queries/v4/test_queries.json`.

V4 evaluates the current parser and chunking pipeline. Its labels were
annotated against the chunks produced by this pipeline:

- PDF parsing uses MinerU Markdown to preserve document structure, formulas,
  and tables, then maps sections back to physical PDF pages where possible.
- Word parsing processes paragraphs and tables in document order, carries the
  heading path into text and table content, and associates captions with their
  data tables.
- Retrieval filtering removes front matter, tables of contents, references,
  appendices, headers and footers, and footnotes before indexing.
- Text sections use recursive 1,000-character chunks with 100-character
  overlap; tables remain intact as single chunks to preserve row and column
  relationships. Every chunk retains source, page, section-title, type, and
  stable-ID metadata for retrieval and citation.

| Variant | Recall@1 | Hit@1 | Recall@5 | Hit@5 | Recall@10 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense only | 0.256 | 0.511 | 0.667 | 0.844 | 0.811 | 0.956 | 0.639 |
| Hybrid RRF | 0.315 | 0.600 | 0.759 | 0.956 | 0.896 | 0.978 | 0.754 |
| Hybrid RRF + rerank | 0.354 | 0.667 | 0.854 | 0.978 | 0.896 | 0.978 | 0.806 |

Hybrid retrieval improves over dense-only retrieval at every reported cutoff.
Reranking then provides the strongest early ranking and the best overall MRR:
Hit@1 rises from 0.600 to 0.667 and MRR from 0.754 to 0.806. The reranked and
non-reranked hybrid pipelines share the same Top-10 candidate recall and
Hit@10 because reranking changes only their ordering.

### Qwen3 Candidate-Pool Experiment (V4)

Results are stored in `evaluation/results/v4_pool30/comparison.json`, using
the same V4 labels as the baseline above. This repeats the "Expanded
Candidate-Pool Experiment" design (below) with the current default reranker:
`bm25_top_k` and `dense_top_k` stay at 40, and `retrieval.hybrid_top_k` in
`config/config.yaml` is raised from 10 to 30, so Qwen3-Reranker-4B reorders
30 RRF-fused candidates instead of 10 before the Top-1/3/5/10 metrics are
computed. `v1_dense` and `v2a_hybrid` are computed independently of this
setting and are therefore identical to the V4 baseline.

| Variant | Recall@1 | Hit@1 | Recall@5 | Hit@5 | Recall@10 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hybrid RRF + rerank (pool 10, baseline) | 0.354 | 0.667 | 0.854 | 0.978 | 0.896 | 0.978 | 0.806 |
| Hybrid RRF + rerank (pool 30) | 0.346 | 0.644 | 0.856 | 1.000 | 0.969 | 1.000 | 0.798 |

The earlier "Expanded Candidate-Pool Experiment" used the pre-Qwen3 default
reranker (MiniLM-L-6-v2) and found that a larger candidate pool *hurt*
Recall@5 and Hit@5 after reranking. Repeating the same pool expansion with
Qwen3-Reranker-4B shows the opposite: Hit@5 and Hit@10 both reach a perfect
1.000 (up from 0.978), and Recall@10 rises from 0.896 to 0.969. The
regression is confined to the top of the ranking — Recall@1, Hit@1, and MRR
each drop slightly (by 0.007–0.023) as a few queries' best chunk is reordered
from rank 1 to rank 2, not dropped from the results. Since the deployed
pipeline sends `final_top_k = 5` chunks to the LLM rather than only the top
result, the Hit@5/Hit@10 gains matter more for answer quality than the small
Recall@1 regression. MiniLM's negative result for pool expansion does not
generalize to Qwen3-Reranker-4B; `config/config.yaml` now keeps
`hybrid_top_k: 30` as the default based on this comparison.

### Qwen3-Embedding-4B Experiment (V4)

Results are stored in `evaluation/results/v4_qwen3_embed/comparison.json`,
using the same V4 labels. This isolates the embedding model:
`retrieval.hybrid_top_k` stays at 30 and Qwen3-Reranker-4B is unchanged; only
`embedding.provider`/`embedding.model_name` switch from the local
`BAAI/bge-large-en-v1.5` (via `sentence-transformers`) to ScaDS.AI-hosted
`Qwen/Qwen3-Embedding-4B` (via the OpenAI-compatible `/embeddings` endpoint,
see "Embedding Provider" above). Switching embedding models changes the
vector dimension, so `vector_store.collection_name` was also changed to
`doc_chunks_qwen3_embed` to avoid mixing incompatible vectors into the
existing collection.

| Variant | Recall@1 | Hit@1 | Recall@5 | Hit@5 | Recall@10 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense only (bge-large-en-v1.5) | 0.256 | 0.511 | 0.667 | 0.844 | 0.811 | 0.956 | 0.639 |
| Dense only (Qwen3-Embedding-4B) | 0.269 | 0.489 | 0.778 | 0.933 | 0.893 | 0.956 | 0.671 |
| Hybrid RRF (bge-large-en-v1.5) | 0.315 | 0.600 | 0.759 | 0.956 | 0.896 | 0.978 | 0.754 |
| Hybrid RRF (Qwen3-Embedding-4B) | 0.359 | 0.711 | 0.767 | 0.956 | 0.911 | 0.956 | 0.808 |
| Hybrid RRF + rerank (bge-large-en-v1.5, pool 30) | 0.346 | 0.644 | 0.856 | 1.000 | 0.969 | 1.000 | 0.798 |
| Hybrid RRF + rerank (Qwen3-Embedding-4B, pool 30) | 0.346 | 0.644 | 0.807 | 0.956 | 0.933 | 0.978 | 0.780 |

Qwen3-Embedding-4B is a clearly stronger embedding model on its own:
dense-only Recall@5, Hit@5, and Recall@10 all improve substantially (+0.111,
+0.089, +0.081), and MRR rises from 0.639 to 0.671. The pre-rerank Hybrid RRF
stage improves further, most notably Hit@1 (+0.111) and MRR (+0.054).

After Qwen3-Reranker-4B reranks the now-different 30-candidate pool, the
final metrics come out slightly below the bge-large-en-v1.5 baseline: Hit@5
drops from a perfect 1.000 to 0.956, and Recall@5/Recall@10/MRR each fall by
0.02–0.05. Per-query inspection shows this is not a uniform regression — some
queries move up in rank while others move down — consistent with the same
effect seen in the "Expanded Candidate-Pool Experiment": a stronger dense
signal changes which distractors enter the RRF-fused pool, and the
reranker's optimal pool size is coupled to the embedding model rather than a
fixed constant. `hybrid_top_k` was tuned (10 → 30) against the previous
embedding and has not been re-swept for Qwen3-Embedding-4B.

Rejection accuracy improved from 0.800 to a perfect 1.000 (5/5): the query
that previously failed (the model over-generalized a tangential passage
about OCT's general defect-detection capability to the study's binary
classifier) is now correctly refused, likely because the new embedding
retrieves a different, less misleading candidate set for that query.

Given the clear dense-retrieval gains and the rejection-accuracy improvement
— at the cost of a small, non-uniform dip in the final reranked Top-5
metrics that a `hybrid_top_k` re-sweep may recover — Qwen3-Embedding-4B
(`embedding.provider: scadsai`) is now the default in `config/config.yaml`.

### 2,000 / 200 Chunk Baseline

Results are stored in `evaluation/results/v1/comparison.json`.

| Variant | Recall@1 | Hit@1 | Recall@5 | Hit@5 | Recall@10 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense only | 0.380 | 0.533 | 0.696 | 0.822 | 0.837 | 0.911 | 0.652 |
| Hybrid RRF | 0.309 | 0.467 | 0.730 | 0.822 | 0.874 | 0.956 | 0.624 |
| Hybrid RRF + rerank | 0.420 | 0.600 | 0.750 | 0.844 | 0.874 | 0.956 | 0.715 |

### 1,000 / 100 Chunk Configuration

Results are stored in `evaluation/results/v2/comparison.json`, using the labels
in `evaluation/test_queries/v2/test_queries.json`.

| Variant | Recall@1 | Hit@1 | Recall@5 | Hit@5 | Recall@10 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense only | 0.294 | 0.556 | 0.739 | 0.956 | 0.778 | 0.956 | 0.704 |
| Hybrid RRF | 0.302 | 0.578 | 0.741 | 0.933 | 0.869 | 0.956 | 0.728 |
| Hybrid RRF + rerank | 0.335 | 0.667 | 0.759 | 0.933 | 0.869 | 0.956 | 0.768 |

Within this experiment, hybrid retrieval plus Cross-Encoder reranking is the
best configuration. Compared with the previous 2,000-character chunk baseline,
it improves Hit@1
from 0.600 to 0.667, Hit@5 from 0.844 to 0.933, and MRR from 0.715 to 0.768.
Hit@10 remains 0.956. Recall values should be interpreted with care because the
two chunk configurations use different relevant-chunk labels and granularity.

### Expanded Candidate-Pool Experiment

Results are stored in `evaluation/results/v3/comparison.json`, using the same
V2 labels. This experiment increases Dense and BM25 retrieval from 20 to 40
candidates, keeps 30 RRF-fused candidates, and reranks those 30 candidates
before calculating the final Top-1/3/5/10 metrics.

| Variant | Recall@1 | Hit@1 | Recall@5 | Hit@5 | Recall@10 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense only | 0.294 | 0.556 | 0.746 | 0.956 | 0.785 | 0.956 | 0.704 |
| Hybrid RRF | 0.313 | 0.600 | 0.741 | 0.933 | 0.835 | 0.956 | 0.736 |
| Hybrid RRF + rerank | 0.335 | 0.667 | 0.726 | 0.911 | 0.857 | 0.956 | 0.767 |

The larger candidate pool does not improve the reranked result over V2: Hit@1
is unchanged at 0.667, MRR changes from 0.768 to 0.767, and both Recall@5 and
Hit@5 decline. The experiment preserves Hit@10 at 0.956, so the added pool
does not recover the two queries that lack relevant evidence in Top-10.

This is a combined retrieval-depth and reranking-pool experiment, rather than
an isolated reranking-pool ablation. The deeper Dense and BM25 candidate lists
also change the fused RRF ordering. The decline suggests that the additional
lower-ranked candidates introduce distractors that the current Cross-Encoder
does not consistently place below the labeled evidence. Future comparisons
should use a dedicated, fixed Chroma collection to minimize approximate-index
tie variation between runs.

### Qwen3 Reranker Ablation

Results are stored in
`evaluation/results/ablation/qwen3_pool_10/comparison.json`. This is a
controlled reranker comparison against
`evaluation/results/ablation/rerank_pool_10/comparison.json`: both runs use
the same V2 labels, 40 Dense candidates, 40 BM25 candidates, and 10 RRF-fused
candidates. Their Dense-only and Hybrid RRF metrics are identical, so the
differences below isolate the reranker.

| Reranker | Recall@1 | Hit@1 | Recall@3 | Hit@3 | Recall@5 | Hit@5 | Recall@10 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniLM-L-6-v2 | 0.346 | 0.689 | 0.615 | 0.889 | 0.752 | 0.956 | 0.835 | 0.956 | 0.782 |
| Qwen3-Reranker-4B | 0.389 | 0.733 | 0.657 | 0.911 | 0.783 | 0.933 | 0.835 | 0.956 | 0.824 |

Qwen3-Reranker-4B is the strongest configuration for early ranking: it
improves Hit@1 by 0.044 and MRR by 0.042, while Recall@1, Recall@3, and
Recall@5 also increase. Candidate recall is unchanged at Top-10 because the
same 10 RRF candidates are reranked. Qwen3 does not strictly dominate MiniLM:
Hit@5 declines from 0.956 to 0.933 because one query's first relevant chunk is
moved below rank five. Use Qwen3 when first-result quality is the priority, and
retain this Top-5 regression as a target for a larger follow-up evaluation.

V4 no-answer evaluation covers five unanswerable queries. The pipeline refused
four, completed all requests without evaluation failures, and achieved 0.800
rejection accuracy:

| Metric | Value |
|---|---:|
| Rejection accuracy | 0.800 |
| Refused queries | 4 / 5 |
| Evaluation failures | 0 |

## Notes

- `data/chroma_db/` is created automatically when indexing runs.
- Versioned evaluation labels are stored under `evaluation/test_queries/`.
- `evaluation/results/` contains reproducible metric snapshots for README and interview review.
