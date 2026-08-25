# 🔬 Agentic RAG Knowledge Assistant: 2–3 Minute Portfolio Demo Script

This document provides a concise, step-by-step walkthrough to rehearse and present the **Agentic RAG Knowledge Assistant** during technical interviews, portfolio demonstrations, or architecture reviews.

---

## 🎯 Demo Objective

Demonstrate how an **Agentic RAG architecture** (`PLAN → RETRIEVE → AUDIT → BOUNDED RETRY → GENERATE / REFUSE`) solves the critical vulnerabilities of standard naive RAG systems:
1. **Hallucination Prevention**: Replaces unchecked generation with deterministic evidence auditing and controlled refusal.
2. **Autonomous Self-Correction**: Diagnoses retrieval gaps and iteratively reformulates queries across bounded retries.
3. **Inspectable Provenance**: Provides token-level verified quotes, sub-question relationship matrices, and full execution traces.
4. **Production Ingestion Architecture**: Supports dynamic user PDF uploads with SHA-256 content-addressable vector caching while preserving research corpus isolation.

---

## 🚀 Quick Setup & Launch

1. **Activate Environment & Verify Keys**:
   Ensure `.env` contains your `OPENAI_API_KEY` (or configure secrets in Streamlit Cloud).
2. **Launch the Application**:
   ```bash
   streamlit run app.py
   ```
   The UI will open at `http://localhost:8501`.

---

## 🎬 Step-by-Step Demonstration Flow (Total: ~3 Minutes)

```
┌───────────────────────────┬───────────────────────────────────┬──────────────┐
│ Scenario                  │ Query / Action                    │ Target Time  │
├───────────────────────────┼───────────────────────────────────┼──────────────┤
│ 1. Grounded Answer        │ Self-RAG reflection tokens        │ ~30 seconds  │
│ 2. Iterative Retry        │ REALM vs DPR index maintenance    │ ~45 seconds  │
│ 3. Controlled Refusal     │ Quantum fidelity (Out-of-Scope)   │ ~30 seconds  │
│ 4. User PDF Ingestion     │ Upload PDF & Query Page Citations │ ~45 seconds  │
│ 5. False Premise (Opt.)   │ Toolformer PPO RL                 │ ~30 seconds  │
└───────────────────────────┴───────────────────────────────────┴──────────────┘
```

---

### Scenario 1: Grounded Answer & Verifiable Citations (~30s)

* **Goal**: Show that answers are derived from verified source passages, not hallucinations or general LLM memory.
* **Corpus Mode**: `Research Corpus (10 Papers)`
* **Action**: In the sidebar dropdown, select **`1. Answerable: Self-RAG reflection tokens`** and click **Ask Assistant**.
* **What to Show on Screen**:
  1. **Status Badge**: Point out `<span class="badge-sufficient">✓ ANSWER: FULLY GROUNDED & VERIFIED</span>`.
  2. **Grounded Response**: Highlight how the answer explains `[Retrieve]`, `[IsREL]`, `[IsSUP]`, and `[IsUSE]` tokens in natural language.
  3. **Expand Citation [1]**: Expand the citation card to show the verbatim source passage extracted directly from `SelfRAG_Asai_2023`.
  4. **Inspect Trace**: Click **Deep Inspect: Complete Agent Execution Trace** $\rightarrow$ show Tab 3 (*Evidence Auditor*) where each sub-question is verified against candidate chunks.
* **What to Say**:
  > *"In standard RAG, the model generates an answer and we hope it didn't hallucinate. Here, our Evidence Auditor verifies exact character-level quotation spans before the generator is even allowed to synthesize the final output."*

---

### Scenario 2: Search Gap Diagnosis & Bounded Retry (~45s)

* **Goal**: Show genuine agentic self-correction when initial retrieval is incomplete.
* **Corpus Mode**: `Research Corpus (10 Papers)`
* **Action**: Select **`2. Retry: REALM vs DPR index maintenance`** and click **Ask Assistant**.
* **What to Show on Screen**:
  1. **Iterative Self-Correction Trace**: Point out the two columns:
     * **Pass 1**: Verdict was `INSUFFICIENT_RETRY` because initial retrieval retrieved general descriptions of REALM and DPR without the specific asynchronous MIPS index refreshing comparison.
     * **Gap Diagnosis**: The system diagnosed the missing concept and reformulated a focused query.
     * **Pass 2**: Reformulated query executed, retrieved the exact index re-embedding and MIPS maintenance mechanisms from `REALM_Guu_2020` and `DPR_Karpukhin_2020`, and returned `SUFFICIENT`.
  2. **Final Answer**: Displays the comparative breakdown grounded in source evidence.
* **What to Say**:
  > *"When Pass 1 fails to find all required comparative facts, our system doesn't guess. The auditor diagnoses the exact missing information, formulates a targeted query, and executes a bounded retry. You can see the multi-pass progression rendered directly from the execution trace."*

---

### Scenario 3: Hallucination Prevention via Controlled Refusal (~30s)

* **Goal**: Show that when the corpus lacks evidence, the system refuses rather than fabricating plausible answers.
* **Corpus Mode**: `Research Corpus (10 Papers)`
* **Action**: Select **`3. Refusal: Quantum fidelity (Out-of-Scope)`** and click **Ask Assistant**.
* **What to Show on Screen**:
  1. **Status Badge**: Point out `<span class="badge-refuse">🛑 REFUSED: INSUFFICIENT EVIDENCE (HALLUCINATION PREVENTION)</span>`.
  2. **Multi-Pass Exhaustion**: Point out that the system attempted 3 retrieval passes before concluding the information is `DEFINITIVELY_ABSENT`.
  3. **Structured Refusal Explanation**: Highlight how the system lists the exact missing evidence dimensions without outputting fabricated claims or citations.
* **What to Say**:
  > *"In our held-out adversarial evaluation, naive Hybrid RAG hallucinated on 100% of unanswerable questions. Our Agentic RAG system achieved 0/4 hallucinations by enforcing an evidence gate that refuses to answer when evidence cannot be verified."*

---

### Scenario 4: User-Uploaded PDF & Dynamic Ingestion (~45s)

* **Goal**: Show dynamic document parsing, section-aware chunking, page-level provenance, and SHA-256 vector caching.
* **Corpus Mode**: Switch sidebar radio to **`Upload PDF`**.
* **Action**:
  1. Drag and drop any research PDF (e.g. `data/raw_documents/CRAG_Yan_2024.pdf` or a sample paper).
  2. Notice the metadata banner: File hash, Page count, Section count, and `✓ PDF Indexed — Embeddings Generated`.
  3. Enter a question specific to the uploaded paper and click **Ask Assistant**.
  4. Expand citation to show **Page-Level Provenance** (`[filename: Page X, §Section]`).
  5. Re-upload or re-select the same file: Notice the immediate `✓ Cache Hit — Embeddings Reused` badge (0 embedding API calls).
* **What to Say**:
  > *"We extended the pipeline to user uploads without cutting corners: uploads are parsed with pdfplumber, section-chunked, and stored in an isolated Content-Addressable cache keyed by SHA-256. Repeat uploads result in 0 embedding API calls, and dynamic queries use the exact same agentic planner and auditor."*

---

### Scenario 5: False Premise Contradiction & Debunking (~30s)

* **Goal**: Show that the system identifies false assumptions in user queries.
* **Corpus Mode**: `Research Corpus (10 Papers)`
* **Action**: Select **`4. Debunk: Toolformer PPO reinforcement learning`** and click **Ask Assistant**.
* **What to Show on Screen**:
  1. **Status Badge**: `<span class="badge-debunk">🛡️ FALSE PREMISE DEBUNKED & CORRECTED</span>` (or `<span class="badge-partial">⚠ PARTIAL ANSWER</span>`).
  2. **Auditor Finding**: The trace notes that Toolformer does *not* use PPO RL, but rather self-supervised in-context API filtering.
  3. **Cited Correction**: Explains the true mechanism with verified citations.

---

## 💡 Key Architectural Talking Points for Interviews

1. **Why Plain Single-Pass RAG Was Insufficient**: Traditional top-$k$ RAG feeds retrieved passages directly to the generator without validation, causing silent hallucinations when context is irrelevant or the question is out of scope.
2. **Why the Evidence Sufficiency Checker + Bounded Retry Loop is the Core Agentic Component**: Query planning alone is open-loop. What makes the system agentic is the closed-loop Evidence Auditor acting as an evaluation gate that verifies atomic claim support, diagnoses retrieval gaps, and controls bounded query reformulations.
3. **Why Retrieval and Generation Were Evaluated Separately**: Measuring cumulative evidence recall ($52.1\%$) separately from answer accuracy ($45.8\%$) and hallucination ($0/4$) decoupled retrieval recall from generator grounding failures.
4. **Why the Held-Out Result Showed a Real Tradeoff Rather than "Agentic RAG is Better"**: Agentic RAG eliminated hallucinations on adversarial unanswerables ($0/4$ vs. $4/4$ for Hybrid RAG), but conservative quote verification reduced answer accuracy ($45.8\%$ vs. $60.4\%$) and increased latency ($\sim 17.7\text{s}$ vs. $\sim 1.86\text{s}$).
5. **Why the Evidence Auditor Became the Main Latency/Cost Bottleneck**: Profiling revealed that the LLM auditor accounted for $\sim 88.2\%$ of total execution latency and $\sim 81.4\%$ of benchmark token spend, motivating future work on two-tier lexical gating.
6. **How Benchmark Isolation, Caching, Mocking, and Budget Controls Improved Reproducibility**: SHA-256 CAS vector caching, query caching, mocked Pydantic test boundaries ($87$ offline tests), and hard execution guards enabled deterministic evaluation with zero accidental API spend.

---

## 🛠️ Fallback & Recovery Guidance

* **If OpenAI API experiences transient latency**:
  * Point out the **Trace Latency breakdown** in Tab 4 to explain component timings.
* **If API key is missing or quota exceeded**:
  * The UI catches `OpenAIAPIDisabledError` gracefully and displays a clear notification without crashing the application.
