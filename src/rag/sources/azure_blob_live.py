"""Live Azure Blob source — a custom CocoIndex LiveMapView.

CocoIndex 1.0.20 ships `azure_blob.list_blobs()` as a one-shot walker with no
live mode (verified in the installed library: `AzureBlobWalker` has no
`live=` parameter and returns no `LiveMapView`), so continuous monitoring is
implemented here: poll the container on an interval and diff **ETags** to
classify each blob as added, modified, or deleted, then push only those
deltas to the subscriber. CocoIndex's memoization means an unchanged ETag
costs nothing downstream.

Change detection reuses CocoIndex's built-in `FileLike.content_fingerprint()`,
which already prefers the backend-supplied ETag (via
`FileMetadata.content_fingerprint`, populated by the connector from the blob
listing) and only falls back to hashing content — so no custom ETag handling
is needed here.

Shape follows `cocoindex.connectors.localfs._source._LiveDirItems`, the
reference `LiveMapView` implementation: initial full scan -> `update_all()`
-> `mark_ready()` -> loop delivering incremental deltas.

Production alternative: Blob -> Event Grid -> queue, which replaces polling
with push notification. The subscriber contract below is identical either
way.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class LiveBlobItems:
    """Satisfies CocoIndex's LiveMapView protocol: __aiter__ + watch()."""

    def __init__(self, walker: Any, *, poll_seconds: float) -> None:
        self._walker = walker
        self._poll_seconds = poll_seconds

    def __aiter__(self):
        return self._walker.items()

    async def _snapshot(self) -> dict[str, tuple[bytes, Any]]:
        return {
            key: (await item.content_fingerprint(), item)
            async for key, item in self._walker.items()
        }

    async def watch(self, subscriber: Any) -> None:
        known = await self._snapshot()
        await subscriber.update_all()
        await subscriber.mark_ready()
        logger.info("blob watcher ready: tracking %d blobs", len(known))

        while True:
            await asyncio.sleep(self._poll_seconds)
            try:
                current = await self._snapshot()
            except Exception:
                logger.exception("blob poll failed; retrying next interval")
                continue

            for key, (fingerprint, item) in current.items():
                previous = known.get(key)
                if previous is None or previous[0] != fingerprint:
                    logger.info(
                        "blob %s: %s",
                        "added" if previous is None else "modified",
                        key,
                    )
                    await subscriber.update(key, item)

            for key in known.keys() - current.keys():
                logger.info("blob deleted: %s", key)
                await subscriber.delete(key)

            known = current


def live_list_blobs(
    container_client: Any,
    *,
    prefix: str = "",
    path_matcher: Any = None,
    poll_seconds: float | None = None,
) -> LiveBlobItems:
    from cocoindex.connectors import azure_blob

    from rag.config import get_settings

    walker = azure_blob.list_blobs(
        container_client, prefix=prefix, path_matcher=path_matcher
    )
    interval = poll_seconds if poll_seconds is not None else get_settings().blob_poll_seconds
    return LiveBlobItems(walker, poll_seconds=interval)
