"""
Configuration settings and constants for the Agentic RAG Knowledge Assistant.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_documents"
PROCESSED_DIR = DATA_DIR / "processed_chunks"
BENCHMARK_DIR = DATA_DIR

# Ensure directories exist
RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Processed Data Artifacts
CHUNKS_JSON_PATH = PROCESSED_DIR / "chunks.json"
DOCS_JSON_PATH = PROCESSED_DIR / "documents.json"
SQLITE_DB_PATH = PROCESSED_DIR / "metadata.db"
EMBEDDINGS_CACHE_PATH = PROCESSED_DIR / "embeddings_cache.npz"
DOC_EMBEDDINGS_CACHE_PATH = PROCESSED_DIR / "document_embeddings.npz"
DOC_EMBEDDINGS_METADATA_PATH = PROCESSED_DIR / "document_embeddings_metadata.json"
QUERY_EMBEDDINGS_CACHE_PATH = PROCESSED_DIR / "query_embeddings_cache.npz"

# Safety & Execution Modes
OPENAI_API_ENABLED = os.getenv("OPENAI_API_ENABLED", "true").lower() in ("true", "1", "yes")
CACHE_ONLY_MODE = os.getenv("CACHE_ONLY_MODE", "false").lower() in ("true", "1", "yes")

# Benchmark Budget Guard: hard limit in USD. Benchmark refuses to run if estimated cost exceeds this.
# Override via MAX_EXPERIMENT_COST_USD environment variable.
MAX_EXPERIMENT_COST_USD: float = float(os.getenv("MAX_EXPERIMENT_COST_USD", "1.00"))

# Chunking Hyperparameters
TARGET_CHUNK_TOKENS = 450
CHUNK_OVERLAP_TOKENS = 75
MIN_CHUNK_TOKENS = 80
TOKENIZER_ENCODING = "cl100k_base"

# Models Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536

# Official Pricing Constants (per 1,000,000 tokens)
EMBEDDING_COST_PER_1M = 0.02
GPT4O_MINI_PROMPT_COST_PER_1M = 0.15
GPT4O_MINI_COMPLETION_COST_PER_1M = 0.60

# Retrieval & Search Hyperparameters
DEFAULT_TOP_K = 5
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60  # Reciprocal Rank Fusion constant

# Agentic Sufficiency & Retry Bounds
MAX_RETRIES = 2
SIMILARITY_NOISE_FLOOR = 0.38  # Empirically tuned on dev set

# Paper Corpus Registry
CORPUS_REGISTRY = [
    {
        "doc_id": "RAG_Lewis_2020",
        "arxiv_id": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": "Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, et al.",
        "year": 2020,
        "venue": "NeurIPS 2020",
        "focus": "Foundations of RAG-Sequence and RAG-Token non-parametric memory models."
    },
    {
        "doc_id": "REALM_Guu_2020",
        "arxiv_id": "2002.08909",
        "title": "REALM: Retrieval-Augmented Language Model Pre-Training",
        "authors": "Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, Ming-Wei Chang",
        "year": 2020,
        "venue": "ICML 2020",
        "focus": "Neural retriever pre-training with unsupervised masked language modeling."
    },
    {
        "doc_id": "DPR_Karpukhin_2020",
        "arxiv_id": "2004.04906",
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "authors": "Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, et al.",
        "year": 2020,
        "venue": "EMNLP 2020",
        "focus": "Dual-encoder dense retrieval architecture and in-batch negative training."
    },
    {
        "doc_id": "ReAct_Yao_2022",
        "arxiv_id": "2210.03629",
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "authors": "Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao",
        "year": 2022,
        "venue": "ICLR 2023",
        "focus": "Interleaving reasoning traces (thoughts) with task-specific actions and observations."
    },
    {
        "doc_id": "SelfRAG_Asai_2023",
        "arxiv_id": "2310.11511",
        "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "authors": "Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi",
        "year": 2023,
        "venue": "ICLR 2024",
        "focus": "Dynamic selective retrieval and reflection tokens ([Retrieve], [IsREL], [IsSUP], [IsUSE])."
    },
    {
        "doc_id": "Toolformer_Schick_2023",
        "arxiv_id": "2302.04761",
        "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
        "authors": "Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, et al.",
        "year": 2023,
        "venue": "NeurIPS 2023",
        "focus": "Self-supervised API call insertion (Calculator, QA, Search, Wiki, Machine Translation)."
    },
    {
        "doc_id": "FLARE_Jiang_2023",
        "arxiv_id": "2305.06983",
        "title": "Active Retrieval Augmented Generation",
        "authors": "Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, et al.",
        "year": 2023,
        "venue": "EMNLP 2023",
        "focus": "Forward-looking active retrieval triggered on low-probability generated tokens."
    },
    {
        "doc_id": "Reflexion_Shinn_2023",
        "arxiv_id": "2303.11366",
        "title": "Reflexion: Language Agents with Verbal Reinforcement Learning",
        "authors": "Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao",
        "year": 2023,
        "venue": "NeurIPS 2023",
        "focus": "Self-reflective memory buffer converting scalar environment feedback into verbal summaries."
    },
    {
        "doc_id": "MemGPT_Packer_2023",
        "arxiv_id": "2310.08560",
        "title": "MemGPT: Towards LLMs as Operating Systems",
        "authors": "Charles Packer, Vivian Fang, Shishir G. Patil, Kevin Lin, Sarah Wooders, Joseph E. Gonzalez",
        "year": 2023,
        "venue": "arXiv 2023",
        "focus": "Hierarchical memory management with main context, external storage, and memory interrupts."
    },
    {
        "doc_id": "CRAG_Yan_2024",
        "arxiv_id": "2401.15884",
        "title": "Corrective Retrieval Augmented Generation",
        "authors": "Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling",
        "year": 2024,
        "venue": "arXiv 2024",
        "focus": "Lightweight retrieval evaluator with confidence thresholds (Correct, Incorrect, Ambiguous)."
    }
]
