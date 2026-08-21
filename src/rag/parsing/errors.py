"""Typed parsing failures.

The pipeline's contract is that one poisoned document must not stop the run
(design §4.1), which means the ingest loop has to be able to tell "this file
is not something we parse" apart from "the process is broken". A bare
`ValueError` cannot carry that distinction: it is also what `int("x")` and a
hundred library internals raise, so catching it either swallows real bugs or
misses real format failures. Every parser-side refusal therefore derives from
`ParsingError`, and the doc_id travels with the exception so the recorded
`IngestError` can name the file without the handler having to reconstruct it.
"""
from __future__ import annotations


class ParsingError(Exception):
    """Base for every failure raised by the parsing layer."""

    def __init__(self, message: str, doc_id: str = "") -> None:
        super().__init__(message)
        self.doc_id = doc_id


class UnsupportedFormatError(ParsingError):
    """No parser available for this format under the current `DOC_PARSER` mode.

    Note "under the current mode": images and unknown formats are perfectly
    supported when Document Intelligence is reachable, and unsupported when it
    is not. The mode is part of the message for that reason.
    """

    def __init__(self, message: str, doc_id: str = "", file_format: str = "") -> None:
        super().__init__(message, doc_id)
        self.file_format = file_format


class ScannedDocumentError(UnsupportedFormatError):
    """A PDF with no text layer reached a path that cannot OCR it.

    Raised instead of letting `LocalParser` run and return an empty block
    list. An empty document looks like a successfully parsed one to every
    downstream stage, so it would be indexed as a document that contains
    nothing -- silently unanswerable rather than visibly failed. This is the
    "never silently produce garbage" rule made enforceable.
    """


class MalformedDocumentError(ParsingError):
    """The bytes claim a format but do not parse as it (truncated, corrupt)."""
