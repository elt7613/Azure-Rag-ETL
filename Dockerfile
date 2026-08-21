# One image, two processes.
#
# The API server and the ETL watcher are the same code with different entry
# points, so they are the same image with different commands (see
# docker-compose.yml). Building them separately would let the retrieval side
# and the ingestion side drift onto different versions of the parsing and
# chunking code, which is exactly the bug that produces answers citing chunks
# that no longer exist.
#
#   API:  uvicorn rag.api.main:app            (the default CMD)
#   ETL:  cocoindex --app-dir . update rag.etl.app --live

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies resolve from pyproject alone, so this layer survives every
# change to src/ and the rebuild stays in the seconds.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

# The shipped corpus, so DOC_SOURCE=local works in the container without a
# mount. A real deployment sets DOC_SOURCE=blob and never reads this.
COPY source_data ./source_data
COPY eval ./eval

# Everything the pipeline persists locally lives under data/: CocoIndex's
# incremental state, the extraction cache and the document registry. It is a
# volume because losing it does not lose your documents, but it does mean the
# next run re-extracts and re-embeds everything -- billable work, and slow.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data \
    && chown -R app:app /app
USER app
VOLUME ["/app/data"]

EXPOSE 8000

# Deliberately not /health: that probe calls Azure, and a liveness check should
# not fail -- or bill -- because a dependency is briefly unavailable. This one
# answers from configuration alone, so it means "the process is up and serving".
# Use GET /health for readiness, where a dependency failure genuinely matters.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/departments', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
