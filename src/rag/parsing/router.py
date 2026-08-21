"""The universal parser router (design §4.1).

The brief is that the system handles *any* document, clean or messy. That is
a routing problem before it is a parsing problem: the right parser depends on
the format, on what is actually inside the bytes, and on which services the
deployment is allowed to call. This module makes that one decision and
nothing else, so it can be reasoned about and tested without parsing anything.

Three ideas carry the design.

**Cheap first, service second.** Every format a local library reads well goes
to a local library. Document Intelligence is reserved for what genuinely needs
OCR or has no local reader at all, because it is a per-page paid call: at 5M
documents the difference between "DI for everything" and "DI where it is
needed" is the difference between a feasible and an infeasible bill.

**The name is a hint, the bytes are the truth.** A scanned PDF and a
born-digital PDF share an extension and need completely different treatment,
and blob stores are full of files with no extension at all. So the extension
is consulted first (it is free and usually right) and the content is consulted
whenever the extension is missing, unknown, or -- for PDFs -- insufficient.

**`DOC_PARSER` is a capability statement, not a parser name.** `local`
promises "this deployment has no Document Intelligence resource"; it still
routes per format, because `PlainParser` and `PptxParser` are local too. What
it cannot serve it refuses with a typed error rather than handing the bytes to
a parser that will return an empty document -- an empty document is
indistinguishable from a successfully-parsed one downstream, so it would be
indexed as a document that answers nothing, and nobody would ever know.
"""
from __future__ import annotations

import io
import logging

from rag.config import Settings, get_settings
from rag.models import ParsedDocument
from rag.parsing.base import DocumentParser
from rag.parsing.errors import ScannedDocumentError, UnsupportedFormatError
from rag.parsing.plain import PLAIN_FORMATS, PlainParser
from rag.parsing.pptx import PptxParser

logger = logging.getLogger(__name__)

#: Formats `LocalParser` reads directly.
LOCAL_FORMATS = frozenset({"pdf", "docx", "xlsx"})
#: Raster formats: no text layer exists, so OCR is the only option.
IMAGE_FORMATS = frozenset({"png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif", "webp"})

# Leading bytes that identify a format regardless of what the file is called.
# Ordered so that longer, more specific signatures are tested first.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"%PDF", "pdf"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"GIF8", "gif"),
    (b"BM", "bmp"),
    (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"),
)

# OOXML files are all ZIP containers, so the magic bytes only get us as far as
# "a zip"; which Office format it is shows up in the part names inside.
_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_OOXML_MARKERS: tuple[tuple[str, str], ...] = (
    ("ppt/", "pptx"),
    ("word/", "docx"),
    ("xl/", "xlsx"),
)

# Formats whose magic bytes are conclusive. A file whose *content* is one of
# these is one of these, whatever it happens to be called -- which is how a
# renamed or wrongly-exported file (a PDF saved as `.docx`, a JPEG saved as
# `.png`) still gets parsed instead of failing on the parser its name asked
# for. Weak signals (HTML, JSON) are deliberately excluded: they are prefix
# guesses, and a markdown file that opens with `{` is still markdown.
_CONCLUSIVE = frozenset({"pdf", "png", "jpg", "gif", "bmp", "tiff",
                         "pptx", "docx", "xlsx"})
# Spellings of the same format, so a `.jpeg` is never "corrected" to `.jpg`.
_ALIASES = {"jpeg": "jpg", "tif": "tiff", "htm": "html", "text": "txt",
            "markdown": "md", "mdown": "md", "mkd": "md", "xhtml": "html"}

# How much of a text-ish file to inspect when guessing HTML/JSON.
_SNIFF_WINDOW = 4096

# Only the first pages are measured for text density. A 200-page scan and a
# 200-page report differ decisively on page one, and opening every page costs
# real time on a corpus this size -- the document is opened again by the
# parser that wins, so this measurement is pure overhead and is kept small.
_DENSITY_SAMPLE_PAGES = 5


# --------------------------------------------------------- format sniffing


def _zip_format(data: bytes) -> str:
    """Which OOXML format a ZIP container holds, or "zip" if it holds none."""
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
    except (zipfile.BadZipFile, OSError):
        return "zip"
    for prefix, file_format in _OOXML_MARKERS:
        if any(name.startswith(prefix) for name in names):
            return file_format
    return "zip"


def _sniff_content(data: bytes) -> str:
    for signature, file_format in _MAGIC:
        if data.startswith(signature):
            return file_format
    if data.startswith(_ZIP_MAGIC):
        return _zip_format(data)

    head = data[:_SNIFF_WINDOW].lstrip()
    lowered = head[:512].lower()
    if lowered.startswith((b"<!doctype html", b"<html", b"<?xml")) or b"<html" in lowered:
        return "html"
    if head[:1] in (b"{", b"["):
        return "json"
    return ""


def sniff_format(doc_id: str, data: bytes) -> str:
    """The canonical format token for a document.

    The extension wins when it names a format we have a parser for -- it is
    free, it is what the author intended, and content sniffing cannot tell
    `.csv` from `.txt` anyway. Content overrules it in the two cases where the
    name is actively wrong: when there is no usable extension at all (a blob
    called `export`), and when the bytes carry a conclusive signature for a
    different format than the name claims.
    """
    tail = doc_id.rsplit("/", 1)[-1]
    suffix = tail.rsplit(".", 1)[-1].lower() if "." in tail else ""
    known = LOCAL_FORMATS | IMAGE_FORMATS | PLAIN_FORMATS | {"pptx"}
    if suffix not in known:
        return _sniff_content(data) or suffix
    sniffed = _sniff_content(data)
    if sniffed in _CONCLUSIVE and _ALIASES.get(suffix, suffix) != sniffed:
        logger.info("%s: content is %s despite the .%s extension; parsing it "
                    "as %s", doc_id, sniffed, suffix, sniffed)
        return sniffed
    return suffix


# ---------------------------------------------------- scanned-PDF detection


def pdf_char_density(data: bytes) -> float:
    """Average extractable characters per page over the sampled pages.

    Kept pure (bytes in, number out) and separate from the routing decision so
    the threshold can be reasoned about against real documents: the corpus's
    born-digital pages run 1,300-1,900 characters, a scan runs 0. There is no
    middle ground to agonise over, which is why a simple average beats
    anything cleverer here.

    Bytes that will not open as a PDF return 0.0. That is not an error path:
    a file pdfplumber cannot open is precisely one that should be handed to
    Document Intelligence, which is far more tolerant of malformed PDFs.
    """
    import pdfplumber

    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = pdf.pages[:_DENSITY_SAMPLE_PAGES]
            if not pages:
                return 0.0
            total = sum(len((page.extract_text() or "").strip()) for page in pages)
            return total / len(pages)
    except Exception:  # noqa: BLE001 - any failure to read means "not readable here"
        logger.debug("pdf_char_density: unreadable PDF, treating as scanned",
                     exc_info=True)
        return 0.0


def is_scanned_pdf(data: bytes, threshold: int) -> bool:
    """Whether this PDF needs OCR, per the configured per-page character floor."""
    return pdf_char_density(data) < threshold


# ------------------------------------------------------------------ router


class _Relabeled:
    """Runs a suffix-dispatching parser under a corrected file name.

    `LocalParser` decides what to do from `doc_id`'s extension, so once the
    router has established that a file called `.docx` is really a PDF, handing
    it straight to `LocalParser` would undo that finding. Rather than reach
    into a module this task does not own, the corrected format is expressed
    the only way `LocalParser` reads it -- as the name -- and the document's
    true `doc_id` is restored afterwards, because `doc_id` is the identity key
    every store and every citation is written against and must never change.
    """

    def __init__(self, inner: DocumentParser, file_format: str) -> None:
        self._inner = inner
        self._file_format = file_format

    async def parse(self, data: bytes, doc_id: str) -> ParsedDocument:
        stem = doc_id.rsplit(".", 1)[0] if "." in doc_id.rsplit("/", 1)[-1] else doc_id
        parsed = await self._inner.parse(data, f"{stem}.{self._file_format}")
        parsed.doc_id = doc_id
        return parsed


def _local(doc_id: str, file_format: str) -> DocumentParser:
    from rag.parsing.local_parser import LocalParser

    tail = doc_id.rsplit("/", 1)[-1]
    suffix = tail.rsplit(".", 1)[-1].lower() if "." in tail else ""
    if suffix == file_format:
        return LocalParser()
    return _Relabeled(LocalParser(), file_format)


def _docint(doc_id: str, file_format: str, settings: Settings,
            reason: str) -> DocumentParser:
    """The Document Intelligence branch, guarded by its own configuration.

    Routing to a service that has no endpoint would surface as an SDK
    authentication error three layers down, naming neither the document nor
    the setting that is missing. Failing here instead keeps the diagnosis in
    the message.
    """
    from rag.parsing.azure_docint import AzureDocIntParser

    if not (settings.azure_docint_endpoint and settings.azure_docint_key):
        raise UnsupportedFormatError(
            f"{doc_id}: {reason} needs Azure Document Intelligence, but "
            "AZURE_DOCINT_ENDPOINT/AZURE_DOCINT_KEY are not set "
            "(set them, or accept that this format cannot be ingested)",
            doc_id, file_format,
        )
    return AzureDocIntParser()


def select_parser_for(doc_id: str, data: bytes) -> DocumentParser:
    """Choose the parser for one document, per the design §4.1 routing table.

    Raises `UnsupportedFormatError` (or its `ScannedDocumentError` subclass)
    when no available parser can handle the document under the current
    `DOC_PARSER` mode. Both derive from `ParsingError`, which is what the
    ingest loop catches to record an `IngestError` and move on.
    """
    settings = get_settings()
    mode = settings.doc_parser

    # A hard override: the operator has asked for Document Intelligence on
    # everything, and DI accepts any of these formats' raw bytes.
    if mode == "azure":
        return _docint(doc_id, sniff_format(doc_id, data), settings,
                       "DOC_PARSER=azure")

    file_format = sniff_format(doc_id, data)
    di_available = mode != "local"

    if file_format == "pdf":
        # Measured once and reused for the log line: opening a large PDF is
        # the most expensive thing this router does, and the number is worth
        # recording because it is how the threshold gets tuned on a real
        # corpus rather than guessed.
        density = pdf_char_density(data)
        if density >= settings.scanned_pdf_char_threshold:
            return _local(doc_id, file_format)
        if not di_available:
            raise ScannedDocumentError(
                f"{doc_id}: {density:.0f} extractable characters per page is "
                f"below the scanned threshold of "
                f"{settings.scanned_pdf_char_threshold}, so this PDF needs "
                "OCR, but DOC_PARSER=local forbids Document Intelligence",
                doc_id, file_format,
            )
        logger.info("routing %s to Document Intelligence: %.0f chars/page is "
                    "below the scanned threshold of %d", doc_id, density,
                    settings.scanned_pdf_char_threshold)
        return _docint(doc_id, file_format, settings, "a scanned PDF")

    if file_format in LOCAL_FORMATS:
        return _local(doc_id, file_format)
    if file_format == "pptx":
        return PptxParser()
    if file_format in PLAIN_FORMATS:
        return PlainParser(file_format)

    if file_format in IMAGE_FORMATS:
        if not di_available:
            raise UnsupportedFormatError(
                f"{doc_id}: images carry no text layer and DOC_PARSER=local "
                "forbids Document Intelligence, so there is nothing to read",
                doc_id, file_format,
            )
        return _docint(doc_id, file_format, settings, "an image")

    # Anything else: DI is the catch-all in `auto` because it accepts far more
    # formats than we enumerate, and refusing outright would make "handles any
    # document" false for the long tail this router exists to cover.
    if not di_available:
        raise UnsupportedFormatError(
            f"{doc_id}: no local parser handles '.{file_format}' and "
            "DOC_PARSER=local forbids the Document Intelligence fallback",
            doc_id, file_format,
        )
    return _docint(doc_id, file_format, settings, f"the unrecognised format "
                   f"'.{file_format}'")


class AutoParser:
    """A `DocumentParser` that picks the real parser once it has the bytes.

    This exists so `select_parser()` keeps its format-agnostic signature --
    the ETL calls `await select_parser().parse(data, doc_id)` and never learns
    that routing happened. It holds no state, so it is safe to construct per
    document.
    """

    async def parse(self, data: bytes, doc_id: str) -> ParsedDocument:
        return await select_parser_for(doc_id, data).parse(data, doc_id)
