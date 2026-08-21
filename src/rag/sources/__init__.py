"""Live Azure Blob source selection.

Local source selection stays in `rag.etl.app`, where Task 10's `SOURCE_DIR`
`ContextKey` base-dir mechanism keeps `file.file_path.path` correctly
relative to `local_source_dir` (not the process cwd) -- required for
`extract_metadata`'s department-from-folder inference. This module only
builds the Blob equivalent: a container client and the configured live Blob
source (`rag.sources.azure_blob_live.LiveBlobItems`), which is the genuinely
custom piece (see that module's docstring for why).
"""
from __future__ import annotations

from typing import Any

from cocoindex.resources.file import FilePathMatcher, PatternFilePathMatcher

from rag.config import get_settings
from rag.sources.azure_blob_live import LiveBlobItems, live_list_blobs

__all__ = ["LiveBlobItems", "live_list_blobs", "build_source", "build_container_client"]


def build_container_client() -> Any:
    """Create an (unopened) async Azure Blob container client.

    Callers open it as an async context manager
    (`async with build_container_client() as container:`) so its HTTP
    session lifecycle matches the app's lifespan -- opened once at startup,
    closed once at shutdown, never per call.

    Authenticates with the storage account key (`ContainerClient(...,
    credential=settings.azure_storage_key)`); does NOT use
    `DefaultAzureCredential` / `az login`.
    """
    from azure.storage.blob.aio import ContainerClient

    settings = get_settings()
    return ContainerClient(
        account_url=f"https://{settings.azure_storage_account}.blob.core.windows.net",
        container_name=settings.azure_storage_container,
        credential=settings.azure_storage_key,
    )


def build_source(
    container_client: Any, *, path_matcher: FilePathMatcher | None = None
) -> LiveBlobItems:
    """Return the configured live Blob source for `container_client`.

    Blob keys are used as-is for `doc_id` (no container-name prefix is ever
    part of a blob name), so e.g. `sales/Pricing2026.pdf` in the container
    yields doc_id `sales/Pricing2026.pdf` -- the department-from-first-
    path-segment inference in `extract_metadata` sees the department
    directly, with no extra prefix to strip.
    """
    settings = get_settings()
    matcher = path_matcher or PatternFilePathMatcher(
        included_patterns=settings.included_patterns
    )
    return live_list_blobs(
        container_client,
        prefix=settings.azure_blob_prefix,
        path_matcher=matcher,
    )
