# Agentic RAG Knowledge Assistant

> An evidence-aware RAG system that plans retrieval, audits evidence, retries when necessary, and refuses unsupported questions.

---

## 1. Problem Statement

Standard Retrieval-Augmented Generation (RAG) architectures follow an unvalidated, feed-forward pipeline:

$$\text{Query} \longrightarrow \text{Retrieve Top-}k \longrightarrow \text{Generate Answer}$$

While computationally fast and simple to deploy, this paradigm exhibits a severe structural flaw: **retrieving topically relevant chunks does not guarantee that the evidence factually supports or completely answers the user's question.**

When faced with out-of-scope questions, subtle retrieval failures, or adversarial prompts containing false premises, naive RAG generators frequently fabricate plausible-sounding answers based on model parametric memory or irrelevant retrieved context. 

This project experimentally investigates whether an **autonomous, closed-loop Agentic RAG system** (`PLAN → RETRIEVE → AUDIT → BOUNDED RETRY → GENERATE / REFUSE`) can substantially improve factual grounding and hallucination resistance, while systematically measuring the corresponding tradeoffs in answer coverage, system latency, and API cost.

---

## 2. Why This Is Agentic

An architecture is not "agentic" simply because it includes an LLM prompt that decomposes a query. What makes this system agentic is its **closed-loop feedback mechanism**: an explicit, deterministic **Evidence Auditor** acts as an evaluation gate that measures factual coverage, diagnoses retrieval gaps, reformulates targeted queries, and controls iterative retries before deciding whether generation is legally permitted.

```
                      ┌────────────────────────┐
                      │     User Question      │
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │     Query Planner      │
                      │  (Decompose & Classify)│
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │    Hybrid Retrieval    │
                      │   (Dense + BM25 RRF)   │
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │    Evidence Auditor    │
                      │ (Atomic Claim Verifier)│
                      └───────────┬────────────┘
                                  │
               ┌──────────────────┴──────────────────┐
               │                                     │
               ▼                                     ▼
        [ SUFFICIENT ]                        [ INSUFFICIENT ]
               │                                     │
               │                              (Pass < Max 3)
               │                                     │
               │                                     ▼
               │                           ┌───────────────────┐
               │                           │ Diagnosed Search  │
               │                           │   Gap & Retry     │
               │                           └─────────┬─────────┘
               │                                     │
               │                                     ▼
               │                           ┌───────────────────┐
               │                           │ Hybrid Retrieval  │
               │                           └─────────┬─────────┘
               │                                     │
               │                                     ▼
               │                           ┌───────────────────┐
               │                           │ Evidence Auditor  │
               │                           └─────────┬─────────┘
               │                                     │
               │                      ┌──────────────┴──────────────┐
               │                      │                             │
               │                      ▼                             ▼
               │               [ SUFFICIENT ]             [ DEFINITIVELY ABSENT ]
               │                      │                             │
               ▼                      ▼                             ▼
        ┌──────────────┐       ┌──────────────┐              ┌──────────────┐
        │   Grounded   │       │   Grounded   │              │  Controlled  │
        │  Generation  │       │  Generation  │              │   Refusal    │
        └──────────────┘       └──────────────┘              └──────────────┘
```

The system refuses to answer when evidence is absent, rather than speculating.

---

## 3. Architecture & Repository Structure

```
Agentic-RAG-Knowledge-Assistant/
├── app.py                            # Streamlit root entrypoint
├── src/
│   ├── config.py                     # Global constants, paths, thresholds, and hyperparameters
│   ├── models/
│   │   ├── document.py               # Pydantic models for Document, Section, and Chunk
│   │   ├── retrieval.py              # SearchResult, RetrievalQuery, and RRF models
│   │   └── trace.py                  # Inspectable execution schemas (QueryPlan, AuditResult, AgentTrace)
│   ├── ingestion/
│   │   ├── pdf_loader.py             # Dual-engine PDF extractor (pdfplumber + pypdf) with academic & resume heading detection
│   │   ├── parser.py                 # Structured HTML/PDF paper parser with section boundary extraction
│   │   ├── section_chunker.py        # Section-aware chunking preserving hierarchy and page provenance
│   │   ├── corpus_fetcher.py         # Automated ArXiv / HTML source corpus downloader
│   │   ├── pipeline.py               # Research corpus ingestion and embedding precomputation pipeline
│   │   └── user_pdf_pipeline.py      # Dynamic PDF ingestion with SHA-256 CAS vector caching
│   ├── retrieval/
│   │   ├── embeddings.py             # OpenAI EmbeddingClient with disk caching & token accounting
│   │   ├── bm25_index.py             # Rank-BM25 sparse keyword retriever
│   │   ├── vector_store.py           # In-memory NumPy cosine similarity vector index
│   │   └── hybrid_retriever.py       # Reciprocal Rank Fusion (RRF, k=60) dense/sparse retriever
│   ├── agent/
│   │   ├── planner.py                # Query classifier, entity extractor, sub-question decomposer (papers + user docs)
│   │   ├── evidence_auditor.py       # Atomic claim verification, contradiction check, and search gap diagnosis
│   │   ├── reformulator.py           # Diagnosed gap query reformulation for iterative retries
│   │   ├── grounded_generator.py     # Fluent natural synthesis with clean [1], [2] in-text citations & schema guarantees
│   │   ├── ablations.py              # Isolated pipeline ablations for benchmark reproducibility
│   │   └── orchestrator.py           # Multi-pass loop controller managing state transitions and retries
│   ├── evaluation/
│   │   ├── benchmark_runner.py       # Evaluator across 8 systems (Baselines, Ablations, Full Agentic)
│   │   ├── metrics.py                # Deterministic accuracy, hallucination, refusal, and recall metrics
│   │   └── dataset_loader.py         # Benchmark dataset validation and loader
│   ├── ui/
│   │   └── app.py                    # Streamlit interface with dual-mode selector, natural citations, and inspectable traces
│   └── utils/
│       └── cost_tracker.py           # Thread-safe token tracking and USD cost accounting
├── tests/
│   ├── conftest.py                   # Shared offline mocks and boundary fixtures (0 API calls default)
│   ├── test_embedding_caching_and_cost.py
│   ├── test_evidence_auditing_refinements.py
│   ├── test_phase1_ingestion.py
│   ├── test_phase2_retrieval.py
│   ├── test_phase3_agentic_core.py
│   ├── test_phase4_baselines_ablations.py
│   ├── test_phase5_evaluation.py
│   ├── test_safety_and_cost_controls.py
│   ├── test_ui.py                    # Streamlit UI unit and state isolation tests
│   └── test_user_pdf_ingestion.py    # Dynamic upload and cache lifecycle tests
├── data/
│   ├── benchmark_dataset.json        # 64-question portfolio benchmark (32 dev / 32 test)
│   ├── demo_questions.json           # 5 scripted portfolio demonstration scenarios
│   ├── raw_documents/                # 10 foundation research PDF papers
│   ├── processed_chunks/             # Frozen research corpus (243 chunks & precomputed embeddings)
│   ├── user_uploads/                 # Content-Addressable Storage (CAS) for dynamic uploads
│   └── evaluation_results/           # Raw benchmark logs, question traces, and aggregate metrics
└── scripts/
    ├── run_evaluation_benchmark.py   # Benchmark execution CLI (dry-run guard + real API flag)
    ├── demonstrate_user_pdf_pipeline.py
    └── run_scripted_demo.py          # Scripted 5-scenario demo runner
```

### Research Corpus vs. Dynamic User Corpus

* **Fixed Research Corpus**: 10 seminal AI/ML papers (RAG, Self-RAG, CRAG, FLARE, ReAct, Toolformer, MemGPT, Reflexion, DPR, REALM) partitioned into 243 section-aware chunks. This corpus and its precomputed vector store in `data/processed_chunks/` are **read-only and frozen** to ensure reproducible evaluation.
* **Dynamic User Corpus**: User-uploaded PDFs are fingerprinted via SHA-256 and stored under `data/user_uploads/{doc_hash}/` with isolated indices (`document.json`, `chunks.json`, `embeddings.npz`, `metadata.json`). Dynamic uploads never modify or contaminate the research corpus.

---

## 4. Evaluation Methodology

The architecture was evaluated on a **portfolio-scale, manually reviewed benchmark** designed to test edge cases where standard RAG models fail:

* **Dataset Size**: 64 total curated questions (32 Development split / 32 Held-Out Test split).
* **Test Set Composition (32 Held-Out Questions)**:
  * **Answerable Single-Hop (16 questions)**: Core factual questions resolvable from a single chunk.
  * **Answerable Multi-Hop (8 questions)**: Comparative or synthesizing queries requiring evidence from multiple distinct papers or sections.
  * **Unanswerable / Out-of-Scope (4 questions)**: Topics entirely absent from the corpus (e.g. quantum fidelity, non-existent 2027 models) to evaluate hallucination prevention.
  * **False-Premise Questions (4 questions)**: Queries containing incorrect assumptions (e.g., asking how Toolformer uses PPO reinforcement learning) to test premise verification.
* **Metric Definitions**:
  * **Answer Accuracy**: Human-reviewed correctness score on answerable subset $[0.0, 1.0]$.
  * **Hallucination Rate**: Fraction of unanswerable questions where the system fabricated an answer ($0.0 = \text{perfect refusal}$).
  * **True Refusal Rate**: Fraction of unanswerable questions correctly identified and refused ($1.0 = \text{perfect refusal}$).
  * **False Refusal Rate**: Fraction of legitimately answerable questions that the system improperly refused.
  * **Citation Precision**: Proportion of generated citations supported by exact verbatim quotes.
  * **Cumulative Evidence Recall**: Fraction of ground-truth gold chunks present in the final candidate pool across all retrieval passes.

---

## 5. Baselines and Ablations

To isolate the contribution of each component, eight distinct system configurations were benchmarked under identical test-set conditions:

1. **`Baseline_A_NaiveDense`**: Standard single-pass dense vector retrieval with Top-5 chunks fed directly to the generator.
2. **`Baseline_B_HybridRAG`**: Single-pass hybrid search (Dense + BM25 with Reciprocal Rank Fusion) without planning or auditing.
3. **`Agentic_System_C_Full`**: Complete agentic loop with Query Planning, Hybrid Retrieval, Atomic Evidence Auditing, and Bounded Retries (max 2 retries).
4. **`Ablation_1_NoPlanner`**: Agentic loop without query decomposition or entity extraction; raw query is searched directly.
5. **`Ablation_2_DenseOnlyAgentic`**: Full agentic planning and auditing loop using dense vector search only (no BM25 keyword matching).
6. **`Ablation_3_NoSufficiencyChecker`**: Query planner + hybrid retrieval passes, but bypasses the auditor and always generates an answer.
7. **`Ablation_4_NoRetryLoop`**: Query planner + hybrid retrieval + auditor, but strictly single-pass (no retry loop upon insufficiency).
8. **`Ablation_5_NoDecomposition`**: Query planner generates search keywords without decomposing multi-hop queries into atomic sub-questions.

---

## 6. Final Held-Out Benchmark Results

The following table presents the **frozen pre-fix held-out experimental results** across all 8 configurations on the 32-question test set:

| System Configuration | Answer Accuracy | Hallucination Rate | True Refusal Rate | False Refusal Rate | Citation Precision | Evidence Recall | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Baseline_A_NaiveDense`** | $14.0/24$ (58.3%) | $4/4$ (100.0%) | $0/4$ (0.0%) | $0/24$ (0.0%) | 40.6% | 41.7% | 1.87s |
| **`Baseline_B_HybridRAG`** | $14.5/24$ (60.4%) | $4/4$ (100.0%) | $0/4$ (0.0%) | $0/24$ (0.0%) | 43.8% | 45.8% | 1.86s |
| **`Ablation_1_NoPlanner`** | $8.5/24$ (35.4%) | $0/4$ (0.0%) | $4/4$ (100.0%) | $3/24$ (12.5%) | 46.5% | 45.8% | 12.43s |
| **`Ablation_2_DenseOnlyAgentic`** | $6.0/24$ (25.0%) | $0/4$ (0.0%) | $4/4$ (100.0%) | $8/24$ (33.3%) | 35.5% | 43.8% | 22.20s |
| **`Ablation_3_NoSufficiencyChecker`** | $17.5/24$ (72.9%) | $4/4$ (100.0%) | $0/4$ (0.0%) | $0/24$ (0.0%) | 40.6% | 50.0% | 3.46s |
| **`Ablation_4_NoRetryLoop`** | $5.5/24$ (22.9%) | $0/4$ (0.0%) | $4/4$ (100.0%) | $9/24$ (37.5%) | 35.3% | 43.8% | 9.24s |
| **`Ablation_5_NoDecomposition`** | $8.0/24$ (33.3%) | $0/4$ (0.0%) | $4/4$ (100.0%) | $1/24$ (4.2%) | 59.5% | 37.5% | 14.19s |
| **`Agentic_System_C_Full`** | $11.0/24$ (45.8%) | **$0/4$ (0.0%)** | **$4/4$ (100.0%)** | $3/24$ (12.5%) | **55.2%** | **52.1%** | 17.69s |

*Note: These numbers represent the frozen baseline evaluation conducted prior to post-evaluation bug fixes.*

---

## 7. Main Finding & The Engineering Tradeoff

The experimental hypothesis was **Partially Supported**:

> **The evidence-aware agentic loop substantially improved resistance to unsupported answers on the adversarial unanswerable subset (0/4 hallucinations vs. 4/4 hallucinations for Hybrid RAG), but introduced a meaningful reduction in answer accuracy (45.8% vs. 60.4%) and an ~8.5× latency increase (~17.7s vs. ~1.86s).**

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 THE CORE RAG TRADEOFF                                  │
├───────────────────────────────────────────┬────────────────────────────────────────────┤
│ Baseline Hybrid RAG                       │ Full Agentic RAG                           │
├───────────────────────────────────────────┼────────────────────────────────────────────┤
│ • Fast execution (~1.86s)                 │ • Substantial latency overhead (~17.7s)    │
│ • Higher answer coverage (60.4% accuracy) │ • Conservative evidence gate (45.8% acc)   │
│ • 100% hallucination on unanswerables     │ • 0% hallucination on adversarial subset   │
│ • Low API token cost ($0.0004/query)      │ • Auditor dominates cost ($0.0091/query)   │
│ • Silent failure mode on missing data     │ • Explicit, controlled refusal mode        │
└───────────────────────────────────────────┴────────────────────────────────────────────┘
```

### Latency and Cost Bottleneck Analysis
Profiling revealed that the **Evidence Auditor accounted for ~88.2% of total execution latency** (15.60s out of 17.69s average) and **~81.4% of total benchmark API spend** ($0.237 of $0.291 total benchmark spend). The Query Planner and Hybrid Retrieval combined accounted for less than 12% of total runtime.

---

## 8. Failure Analysis

Analyzing the benchmark failures exposed three distinct structural failure modes:

### 1. False-Premise Propagation (Q38)
* **Question**: *"How does the RAG-Sequence model perform approximate MIPS scoring using BM25 indices during inference?"*
* **Symptom**: The system attempted to answer how RAG-Sequence uses BM25 instead of recognizing that RAG-Sequence uses dense MIPS search.
* **Root Cause**: The query planner generated keyword queries that retrieved general RAG-Sequence passages and BM25 baseline passages from the same paper. The auditor verified evidence for RAG-Sequence and evidence for BM25 separately without explicitly cross-checking whether the user's premise was valid.
* **Post-Evaluation Fix**: Introduced an explicit `contradiction_check` prompt phase and dedicated `DEBUNK_FALSE_PREMISE` routing in the auditor.
* **Remaining Limitation**: When false premises involve subtle domain nomenclature not explicitly contradicted in text, the auditor may fall back to `PARTIALLY_SUFFICIENT` rather than definitive debunking.

### 2. Numerical Scientific-Notation Verification (Q33)
* **Question**: *"What convergence threshold and learning rate are reported for Agent-Synthesizer?"*
* **Symptom**: The auditor rejected a valid retrieved chunk containing `1e-5` because the generator prompt expected `10^-5` or `0.00001`.
* **Root Cause**: Strict character-exact quotation matching failed on trivial LaTeX/scientific formatting differences.
* **Post-Evaluation Fix**: Implemented a conservative, AST-based numerical equivalence normalizer (`1e-5` $\equiv$ `10^-5` $\equiv$ `0.00001`) that preserves source grounding without hallucinating ungrounded values.
* **Remaining Limitation**: Complex mathematical derivations and multidimensional matrix notation remain beyond conservative regex normalization.

### 3. Multi-Hop Retrieval Fragmentation (Q30)
* **Question**: *"How does Self-RAG's segment-level beam search compare with standard greedy decoding in RAG?"*
* **Symptom**: The system executed 3 retrieval passes and ultimately refused the question.
* **Root Cause**: While Self-RAG passages were retrieved in Pass 1, standard RAG decoding passages were not ranked in the top-5 during Pass 2. The auditor correctly refused to answer rather than guessing standard RAG decoding mechanics.
* **Mitigation**: Confirms the safety gate is functioning as designed, while pointing to the need for cross-document entity linking during multi-hop query planning.

---

## 9. Dynamic User PDF Ingestion

The application includes an isolated dynamic ingestion pipeline for arbitrary user-uploaded documents (research papers, technical whitepapers, resumes, and reports):

```
[User Uploads PDF]
       │
       ▼
1. PDF Parsing (`src/ingestion/pdf_loader.py`)
   - Primary extractor: `pdfplumber` (page numbers, section headings)
   - Fallback extractor: `pypdf`
   - Semantic Heading Recognition: Detects academic sections (Abstract, Methods, Results, Discussion) and career sections (Experience, Skills, Education, Projects, Certifications)
   - Content verification: detects blank/scanned files (`UnextractablePDFError`)
       │
       ▼
2. SHA-256 Fingerprinting & CAS Caching (`src/ingestion/user_pdf_pipeline.py`)
   - Directory: `data/user_uploads/{sha256_hash}/`
   - First upload (Cache Miss): Chunks, generates dense embeddings, persists `embeddings.npz`
   - Repeat upload (Cache Hit): Reuses vector matrix (**0 document embedding API calls**)
       │
       ▼
3. Isolated Hybrid Retrieval (`src/retrieval/hybrid_retriever.py`)
   - Builds in-memory BM25 + dense index specific to the uploaded document
   - 100% isolated from `data/processed_chunks/`
       │
       ▼
4. Unified Agentic Execution (`src/agent/orchestrator.py`)
   - Dynamic document is queried through identical Planner → Retriever → Auditor pipeline
   - Output includes fluent natural-language synthesis, inline `[1], [2]` citation tags, and verified source expanders
```

---

## 10. Scripted Portfolio Demonstrations & Curated Question Bank

See [`DEMO.md`](DEMO.md) for a comprehensive 2–3 minute rehearsal guide. The repository includes 5 verified demonstration scenarios defined in [`data/demo_questions.json`](data/demo_questions.json):

1. **Scenario 1 — Grounded Answer**: *"How does Self-RAG use reflection tokens during inference?"*
   * *Outcome*: Pass 1 `SUFFICIENT` $\rightarrow$ Grounded synthesis with verbatim quotes from `SelfRAG_Asai_2023`.
2. **Scenario 2 — Dynamic Retry**: *"Compare the retrieval index maintenance in REALM versus DPR."*
   * *Outcome*: Pass 1 `INSUFFICIENT_RETRY` $\rightarrow$ Diagnosed gap $\rightarrow$ Pass 2 `SUFFICIENT` $\rightarrow$ Grounded comparative answer.
3. **Scenario 3 — Controlled Refusal**: *"What is the quantum teleportation fidelity achieved by Agent-Q in 2026?"*
   * *Outcome*: Pass 1, 2, 3 exhausted $\rightarrow$ `REFUSE` with structured missing evidence explanation (0 hallucinations).
4. **Scenario 4 — User PDF Upload**: Upload `Agent_Synthesizer_2026.pdf` $\rightarrow$ Query convergence threshold $\rightarrow$ Page-level citation (`Page 3`). Re-upload demonstrates instant `CACHE HIT (0 API calls)`.
5. **Scenario 5 — False Premise Handling**: *"How does Toolformer use Proximal Policy Optimization (PPO) reinforcement learning to optimize API calls?"*
   * *Outcome*: Detects unverified PPO premise $\rightarrow$ Returns `PARTIALLY_SUFFICIENT` citing Toolformer's true self-supervised filtering mechanism.

To execute all 5 scenarios and view the live cost/latency summary:
```bash
python -m scripts.run_scripted_demo
```

### 💡 Curated Questions for Live Exploration

| Category | Recommended Query | Target Behavior & Architecture Proof |
| :--- | :--- | :--- |
| **Factual Single-Hop** | *"How does CRAG use confidence thresholds to trigger Correct, Incorrect, and Ambiguous actions?"* | Hybrid retrieval + exact quote verification (`CRAG_Yan_2024 §4.3`). |
| **Factual Single-Hop** | *"How does FLARE decide when to trigger retrieval dynamically during generation?"* | Active retrieval token confidence tracking (`FLARE_Jiang_2023`). |
| **Multi-Hop Comparative** | *"Compare the retrieval index maintenance in REALM versus DPR."* | Pass 1 retry $\rightarrow$ gap diagnosis $\rightarrow$ Pass 2 dual-paper synthesis. |
| **Multi-Hop Comparative** | *"How do Self-RAG and CRAG differ in how they correct low-quality retrieved passages?"* | Cross-paper evaluation and attribution across reflection tokens vs. external web search. |
| **False-Premise Debunking** | *"How does Toolformer use Proximal Policy Optimization (PPO) reinforcement learning to optimize API calls?"* | False premise detection (`has_conflict=True`) $\rightarrow$ debunks PPO and cites self-supervised loss filtering. |
| **Controlled Refusal** | *"What is the quantum teleportation fidelity achieved by Agent-Q in 2026?"* | Hallucination prevention gate $\rightarrow$ exhausts 3 passes $\rightarrow$ refuses without fabricating. |
| **User Resume / PDF** | *"Summarize the candidate's professional experience and key accomplishments."* | Dynamic PDF extraction with section-aware provenance and clean citation cards. |
| **User Resume / PDF** | *"What programming languages, frameworks, and technical skills are listed?"* | Exact-grounded skill extraction without parametric extrapolation. |

---

## 11. Cost Engineering & Test Isolation

To support rapid development and reproducible continuous integration without incurring API charges:

1. **100% Offline Default Tests**: **87 active offline tests** run with zero external OpenAI API calls. Live integration tests are explicitly separated and opt-in.
2. **Persistent Document Embedding Cache**: Document chunks are hashed and stored in `data/processed_chunks/document_embeddings.npz`. Re-indexing loads from disk in $\le 50\text{ ms}$.
3. **Persistent Query Embedding Cache**: Benchmark evaluation queries are cached in `query_embeddings_cache.npz`, achieving a >92% query cache hit rate.
4. **Execution Guards & Hard Budget Limits**: Benchmark runners require an explicit `--real-api` CLI flag and enforce a hard spend cap ($0.50 default).

---

## 12. Reproducibility & Setup Guide

### 1. Clone and Environment Setup
*Requires Python 3.10+ (tested on Python 3.10, 3.11, 3.12, and 3.14).*

```bash
git clone https://github.com/Rajchhapariya/Agentic-RAG-Knowledge-Assistant.git
cd Agentic-RAG-Knowledge-Assistant

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory (see `.env.example`):
```env
OPENAI_API_KEY=your-actual-openai-api-key
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CACHE_ONLY_MODE=false
OPENAI_API_ENABLED=true
```

### 3. Run Offline Test Suite
```bash
# Runs 87 active offline tests with zero external API calls
pytest

# Optional: Live integration tests (explicit opt-in, consumes API credits)
pytest -m integration
```

### 4. Launch the Streamlit Web Application
```bash
# Run locally
streamlit run app.py
```
Open `http://localhost:8501` to test the research corpus presets or upload custom PDFs.

#### Deploying to Streamlit Community Cloud:
1. Connect repository `Rajchhapariya/Agentic-RAG-Knowledge-Assistant` at [share.streamlit.io](https://share.streamlit.io).
2. Set **Main file path**: `app.py`.
3. In **Advanced Settings → Secrets**, add your credentials:
   ```toml
   OPENAI_API_KEY = "your-actual-openai-api-key"
   OPENAI_MODEL = "gpt-4o-mini"
   OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
   ```
4. Click **Deploy**.

### 5. Run Benchmark (Optional - Consumes API Credits)
```bash
# Dry-run validation (0 API calls, validates dataset integrity)
python -m scripts.run_evaluation_benchmark --split test

# Real benchmark evaluation (explicit opt-in)
python -m scripts.run_evaluation_benchmark --split test --real-api
```

---

## 13. System Limitations

1. **Portfolio-Scale Benchmark**: Evaluated on 64 curated questions across 10 papers; while realistic and adversarial, it is not a 10,000-sample academic benchmark.
2. **Auditor Latency Overhead**: Running an LLM-based Evidence Auditor on every retrieval pass creates an ~8.5× latency penalty over naive RAG.
3. **Absence of OCR**: The PDF parser relies on text streams; image-only or scanned PDFs cannot currently be processed.
4. **False-Premise Complexity**: When an adversarial question introduces multiple nested misconceptions, the system may classify the query as partially sufficient rather than executing a clean debunking.
5. **No Long-Term Memory**: The agent operates per query and does not maintain multi-turn conversational memory.

---

## 14. Proposed Future Work

* **Two-Tier Lexical Pre-Auditing**: Implement lightweight BM25/fuzzy lexical overlap filters to reject obviously insufficient chunks before invoking the expensive LLM auditor.
* **Speculative Parallel Auditing**: Audit candidate chunks asynchronously during multi-query retrieval passes to reduce latency by ~60%.
* **Distilled Local Auditor Model**: Train a lightweight 3B/7B parameter local model fine-tuned specifically for token-level evidence verification.
* **OCR Ingestion Support**: Integrate Tesseract or LayoutLM for scanned PDF support.

---

## 15. Key Engineering Lessons Learned

1. **Retrieval and Verification are Distinct Engineering Problems**: In naive RAG, improving retrieval recall often increases hallucination because the generator receives more irrelevant context. A separate verification gate is required to decouple retrieval volume from output hallucination.
2. **Agentic Loops Require Calibration Against False Refusal**: Introducing an evidence checker easily leads to overly conservative behavior where answerable queries are refused due to minor formatting mismatches. Calibrating quote verification was critical.
3. **Inspectability Trumps "Magical" Black-Box Agents**: Surfacing decomposed sub-questions, individual pass records, exact verified quotes, and component latencies in the UI made debugging and evaluating the system drastically more reliable.
4. **Deterministic Mock Architecture is Essential**: Building realistic Pydantic fixture boundaries allowed us to expand the test suite to 87 tests running in under 60 seconds with zero API spend.

---

## 16. What I Would Discuss in an Interview

1. **Why Plain Single-Pass RAG Was Insufficient**: Traditional top-$k$ RAG feeds retrieved passages directly to the generator without validation, causing silent hallucinations when context is irrelevant or the question is out of scope.
2. **Why the Evidence Sufficiency Checker + Bounded Retry Loop is the Core Agentic Component**: Query planning alone is open-loop. What makes the system agentic is the closed-loop Evidence Auditor acting as an evaluation gate that verifies atomic claim support, diagnoses retrieval gaps, and controls bounded query reformulations.
3. **Why Retrieval and Generation Were Evaluated Separately**: Measuring cumulative evidence recall ($52.1\%$) separately from answer accuracy ($45.8\%$) and hallucination ($0/4$) decoupled retrieval recall from generator grounding failures.
4. **Why the Held-Out Result Showed a Real Tradeoff Rather than "Agentic RAG is Better"**: Agentic RAG eliminated hallucinations on adversarial unanswerables ($0/4$ vs. $4/4$ for Hybrid RAG), but conservative quote verification reduced answer accuracy ($45.8\%$ vs. $60.4\%$) and increased latency ($\sim 17.7\text{s}$ vs. $\sim 1.86\text{s}$).
5. **Why the Evidence Auditor Became the Main Latency/Cost Bottleneck**: Profiling revealed that the LLM auditor accounted for $\sim 88.2\%$ of total execution latency and $\sim 81.4\%$ of benchmark token spend, motivating future work on two-tier lexical gating.
6. **How Benchmark Isolation, Caching, Mocking, and Budget Controls Improved Reproducibility**: SHA-256 CAS vector caching, query caching, mocked Pydantic test boundaries ($87$ offline tests), and hard execution guards enabled deterministic evaluation with zero accidental API spend.

---

## License

This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.
