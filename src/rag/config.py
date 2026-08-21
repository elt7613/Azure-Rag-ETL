"""Single source of truth for all configuration. Nothing else reads os.environ."""
from __future__ import annotations

from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- source selection ----
    doc_source: Literal["local", "blob"] = "local"
    local_source_dir: Path = Path("source_data")
    departments: list[str] = Field(
        default=["HR", "finance", "IT", "legal", "sales"]
    )
    included_patterns: list[str] = Field(
        default=[
            "**/*.pdf", "**/*.docx", "**/*.xlsx", "**/*.pptx",
            "**/*.csv", "**/*.tsv", "**/*.txt", "**/*.md",
            "**/*.html", "**/*.htm", "**/*.json",
            "**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.tiff", "**/*.bmp",
        ]
    )

    # Optional department -> source location mapping, e.g.
    # {"HR": "people-ops/"}. A department absent from this map uses its own
    # name as the path segment / blob prefix. This is what makes adding a
    # department an env edit rather than a code change.
    department_sources: dict[str, str] = Field(default_factory=dict)

    # What to do with a document whose first path segment matches no
    # configured department. "skip" keeps the index scoped to exactly the
    # departments named in DEPARTMENTS; "ingest" files it under its own
    # segment name so nothing is silently lost.
    unknown_department_policy: Literal["skip", "ingest"] = "skip"

    # ---- live monitoring ----
    live_rescan_seconds: int = 300
    blob_poll_seconds: int = 60

    # ---- azure blob ----
    azure_storage_account: str = ""
    azure_storage_container: str = "knowledge-base"
    azure_storage_key: str = ""
    azure_storage_connection_string: str = ""
    azure_blob_prefix: str = ""

    # ---- parsing ----
    # "auto" routes per format and per document: cheap local parsers where they
    # are reliable, Azure Document Intelligence where they are not (scanned
    # PDFs, images, anything the local parsers do not cover). "local" and
    # "azure" remain hard overrides so the pipeline can run with no Document
    # Intelligence resource at all, or force DI for every document.
    doc_parser: Literal["local", "azure", "auto"] = "auto"
    azure_docint_endpoint: str = ""
    azure_docint_key: str = ""

    # A born-digital PDF page yields hundreds of extractable characters; a
    # scanned page yields a handful of stray glyphs or none. Below this
    # per-page average the document is treated as scanned and sent to DI's
    # OCR path instead of pdfplumber.
    scanned_pdf_char_threshold: int = 100

    # Captioning embedded figures costs one vision call per image, so it is
    # off unless asked for. On, charts and diagrams become retrievable.
    vision_captions_enabled: bool = False

    # ---- azure openai ----
    azure_openai_endpoint: str
    azure_openai_key: str
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_embedding_deployment: str
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 32

    # Chat/completion deployment. Kept separate from the embedding API version
    # because the two features move on different Azure preview tracks -- JSON
    # schema structured outputs need a newer api-version than embeddings do.
    azure_openai_chat_deployment: str = "gpt-4.1-mini"
    azure_openai_chat_api_version: str = "2024-12-01-preview"

    # ---- azure openai, batch ----
    # Batch jobs are rejected against a GlobalStandard deployment with
    # `invalid_deployment_type`; they need a `globalbatch` or `datazonebatch`
    # one, which is usually a *separate resource* because a single Azure OpenAI
    # account holds one SKU per model. So the batch path gets its own endpoint,
    # key and deployment, each falling back to the online values when unset --
    # a deployment that happens to serve both needs no extra configuration.
    azure_openai_batch_endpoint: str = ""
    azure_openai_batch_key: str = ""
    azure_openai_batch_deployment: str = ""
    azure_openai_batch_api_version: str = ""

    @property
    def batch_endpoint(self) -> str:
        return self.azure_openai_batch_endpoint or self.azure_openai_endpoint

    @property
    def batch_key(self) -> str:
        return self.azure_openai_batch_key or self.azure_openai_key

    @property
    def batch_deployment(self) -> str:
        return self.azure_openai_batch_deployment or self.azure_openai_chat_deployment

    @property
    def batch_api_version(self) -> str:
        return self.azure_openai_batch_api_version or self.azure_openai_chat_api_version

    # ---- azure ai search ----
    azure_search_endpoint: str
    azure_search_key: str
    azure_search_index: str
    azure_search_semantic_config: str = "default-semantic"

    # ---- cocoindex internal state ----
    # CocoIndex 1.x keeps its incremental state in an embedded LMDB store at a
    # FILESYSTEM PATH (env COCOINDEX_DB), not in Postgres. The v0
    # `COCOINDEX_DATABASE_URL` setting is a deprecated leftover with no effect
    # in v1 -- the library's own setting.py says so -- and is not carried here.
    # `extra="ignore"` above means an older .env that still sets it is fine.
    cocoindex_db: Path = Path("data/cocoindex")

    # ---- neo4j ----
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    graph_enabled: bool = True

    # ---- graph relationship extraction ----
    graph_extraction_enabled: bool = True
    # "online" extracts inline during ingest (right for the incremental
    # deltas live monitoring produces); "batch" routes through the Azure
    # OpenAI Batch API at half price for bulk backfill.
    graph_extract_mode: Literal["online", "batch"] = "online"
    # Units below this many tokens carry no extractable obligation worth an
    # LLM call -- headings, one-line notes, page furniture. Measured against
    # the real corpus rather than guessed: at 40 this dropped
    # IT/PasswordPolicy's account-lockout rule, a genuine policy stated in 38
    # tokens. 30 keeps it without letting furniture back in.
    graph_extract_min_tokens: int = 30
    # Density of entity-bearing signal (proper nouns, money, dates, durations,
    # deontic verbs) a unit must clear before it is worth extracting from.
    graph_extract_min_signal: float = 0.02
    # Small units are packed into one call up to this budget so the fixed
    # prompt overhead is amortised across several of them.
    graph_extract_pack_tokens: int = 3000
    graph_extract_concurrency: int = 8
    extraction_cache_path: Path = Path("data/extraction_cache.db")
    # One row per ingested document, holding what cross-document supersession
    # resolution needs. See rag.targets.version_sync for why it is a sidecar
    # rather than a field on the search index or a read from the graph.
    document_registry_path: Path = Path("data/document_registry.db")
    # A unit hash seen in at least this many distinct documents is boilerplate
    # (headers, footers, standard legal blocks) and is extracted once, not once
    # per document.
    boilerplate_doc_threshold: int = 3
    # Above this similarity two entity mentions are the same entity.
    entity_merge_threshold: float = 0.90

    # ---- retrieval ----
    retrieval_top_k: int = 8
    # Azure's semantic ranker reranks the top 50 RRF candidates, so the vector
    # side must supply at least that many for it to have anything to work with.
    vector_k: int = 50
    rerank_top_k: int = 20
    # Reranker scores run 0-4. Below this, the evidence does not support an
    # answer and the system abstains rather than guessing.
    sufficiency_threshold: float = 1.6
    query_cache_enabled: bool = True
    # Deliberately tight: the documented failure mode of semantic caching is
    # returning one scope's answer to another scope's question.
    semantic_cache_threshold: float = 0.93
    # Departments an API caller gets when the auth gateway supplies no scope.
    # Empty by default and deliberately so: a request that establishes no
    # identity must read nothing. A single-tenant deployment can opt in.
    api_default_departments: list[str] = Field(default_factory=list)
    query_cache_ttl_seconds: int = 900
    query_cache_max_entries: int = 512

    # ---- cost accounting (USD per 1M tokens) ----
    cost_per_1m_input: float = 0.40
    cost_per_1m_cached_input: float = 0.10
    cost_per_1m_output: float = 1.60
    cost_per_1m_embedding: float = 0.02

    # ---- chunking ----
    chunk_target_tokens: int = 700
    chunk_max_tokens: int = 900
    chunk_overlap_tokens: int = 100
    tokenizer_encoding: str = "cl100k_base"


    @model_validator(mode="after")
    def _derive_storage_from_connection_string(self) -> "Settings":
        """Accept either the account+key pair or a full connection string.

        The Azure portal's "Access keys" blade offers both; deriving one from the
        other means a user can paste whichever they copied without it mattering.
        Explicit account/key always win over the connection string.
        """
        conn = self.azure_storage_connection_string
        if conn and not (self.azure_storage_account and self.azure_storage_key):
            parts = dict(
                kv.split("=", 1) for kv in conn.split(";") if "=" in kv
            )
            if not self.azure_storage_account:
                self.azure_storage_account = parts.get("AccountName", "")
            if not self.azure_storage_key:
                # AccountKey is base64 and contains '=' padding, so re-join it.
                key = parts.get("AccountKey", "")
                marker = "AccountKey="
                if marker in conn:
                    key = conn.split(marker, 1)[1].split(";", 1)[0]
                self.azure_storage_key = key
        return self

    @property
    def rescan_interval(self) -> timedelta:
        return timedelta(seconds=self.live_rescan_seconds)


@lru_cache
def get_settings() -> Settings:
    return Settings()
