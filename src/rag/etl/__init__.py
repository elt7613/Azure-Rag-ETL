"""CocoIndex ETL package.

Deliberately re-exports only the callables, NOT the `coco.App` instance. Naming
the App `app` here would shadow the `rag.etl.app` *module*: `from rag.etl
import app` would silently hand back the App object, so anything trying to
import the module -- to patch it, to read INGEST_STATS, to introspect it in a
test -- would get an object with none of those attributes and no error to
explain why. Import the App explicitly from `rag.etl.app` where it is needed.
"""
from rag.etl.app import ingest_one, process_document

__all__ = ["ingest_one", "process_document"]
