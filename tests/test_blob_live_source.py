import asyncio
import pytest


class FakeSubscriber:
    def __init__(self):
        self.updated, self.deleted = [], []
        self.ready = False
        self.full_scans = 0

    async def update_all(self):
        self.full_scans += 1

    async def mark_ready(self):
        self.ready = True

    async def update(self, key, value):
        self.updated.append(key)

    async def delete(self, key):
        self.deleted.append(key)


class FakeFile:
    def __init__(self, etag: str):
        self._etag = etag

    async def content_fingerprint(self) -> bytes:
        return self._etag.encode()


class FakeWalker:
    """Stands in for AzureBlobWalker: yields (key, file) with a mutable etag map."""

    def __init__(self, state):
        self.state = state

    async def items(self):
        for key, etag in list(self.state.items()):
            yield key, FakeFile(etag)


async def test_detects_new_changed_and_deleted_blobs():
    from rag.sources.azure_blob_live import LiveBlobItems

    state = {"HR/a.pdf": "e1", "sales/b.pdf": "e2"}
    live = LiveBlobItems(FakeWalker(state), poll_seconds=0.01)
    sub = FakeSubscriber()

    task = asyncio.create_task(live.watch(sub))
    await asyncio.sleep(0.05)
    assert sub.ready and sub.full_scans == 1

    state["IT/c.pdf"] = "e3"          # new
    state["HR/a.pdf"] = "e1-changed"  # modified
    del state["sales/b.pdf"]          # deleted
    await asyncio.sleep(0.08)
    task.cancel()

    assert "IT/c.pdf" in sub.updated
    assert "HR/a.pdf" in sub.updated
    assert "sales/b.pdf" in sub.deleted


async def test_unchanged_blobs_are_not_reprocessed():
    from rag.sources.azure_blob_live import LiveBlobItems

    live = LiveBlobItems(FakeWalker({"HR/a.pdf": "e1"}), poll_seconds=0.01)
    sub = FakeSubscriber()
    task = asyncio.create_task(live.watch(sub))
    await asyncio.sleep(0.06)
    task.cancel()
    assert sub.updated == []
