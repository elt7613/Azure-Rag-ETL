"""Entity-anchored retrieval over the knowledge graph.

Vector search answers *what text resembles this question*. Some questions are
not that shape:

- *What does the CISO have to approve?* — a traversal from a Role to every
  obligation pointing at it, which may be stated in five documents that share
  no vocabulary with the question.
- *What replaced this policy?* — a `SUPERSEDES` edge, not a similarity.
- *What else does this limit apply to?* — one hop out from an entity.

Dressing those up as similarity searches gets an approximate answer to an exact
question. So the graph is a retriever in its own right, returning the same
`RetrievedChunk` shape as the vector side so fusion can rank the two honestly
against each other.

**How it finds an anchor.** The query is reduced to candidate entity mentions
and matched against `Entity` nodes *within the caller's departments*, in the
Cypher rather than afterwards. From the anchors it walks two hops:

1. Chunks that mention an anchor directly.
2. Chunks that mention an entity one relation away from an anchor — scored
   lower, because relevance decays with distance and a two-hop chunk that
   outranks a direct mention is usually noise.

**What it deliberately does not do.** It does not embed anything, and it does
not rank by text similarity — that is the vector retriever's job, and
duplicating it here would mean fusing two correlated opinions and calling it
agreement.
"""
from __future__ import annotations

import logging
import re

from rag.config import get_settings
from rag.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

# Words that never name an entity. Kept small on purpose: an over-eager stop
# list drops real anchors like "Plan" or "Rate", which are entity types here.
_STOPWORDS = frozenset("""
a about after all am an and any are as at be been before being between both but
by can could did do does doing for from get give had has have how i if in into
is it its just like me more most much my no nor not of off on once only or
other our out over own same should so some such than that the their them then
there these they this those through to too under until up very was we were what
when where which while who whom why will with would you your
each per many much give need needs required must shall may
""".split())

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]+")
# A capitalised run of words mid-sentence is the strongest cheap signal that
# something is a named thing: "Okta Verify", "Enterprise Plus", "Paid Time Off".
_PROPER_RE = re.compile(r"\b([A-Z][A-Za-z0-9'\-]*(?:\s+[A-Z][A-Za-z0-9'\-]*)+)\b")

# Relevance decays with distance. A chunk two hops out is context, not an
# answer, and must not outrank something that mentions the anchor itself.
_DIRECT_SCORE = 1.0
_ONE_HOP_SCORE = 0.45
_SUPERSEDES_SCORE = 0.8
# How many one-hop neighbours may contribute to a single chunk's score.
_MAX_ONE_HOP_CREDITS = 2


def candidate_mentions(query: str, *, max_terms: int = 8) -> list[str]:
    """Phrases from the query worth looking up as entities.

    Multi-word proper phrases first (they are the most specific), then single
    content words. Capped, because each becomes a graph lookup and a long
    question would otherwise fan out into dozens of them.
    """
    mentions: list[str] = []
    for phrase in _PROPER_RE.findall(query):
        words = phrase.split()
        # A sentence-initial capital is not evidence of a name: "Which MFA
        # factor..." would otherwise yield the phrase "Which MFA". Peel
        # stopwords off the front until a real token leads.
        while words and words[0].lower() in _STOPWORDS:
            words.pop(0)
        if len(words) < 2:
            continue
        cleaned = " ".join(words).lower()
        if cleaned not in mentions:
            mentions.append(cleaned)

    for token in _TOKEN_RE.findall(query):
        lowered = token.lower()
        if lowered in _STOPWORDS or len(lowered) < 3:
            continue
        if lowered not in mentions:
            mentions.append(lowered)

    return mentions[:max_terms]


# Anchors are matched on the normalized entity name. `CONTAINS` rather than
# equality so "PTO accrual" finds the "PTO" entity; the department filter and
# the length floor in `candidate_mentions` are what stop that being a scan of
# everything.
_ANCHOR_CYPHER = """
UNWIND $mentions AS mention
MATCH (e:Entity)
WHERE e.department IN $departments
  AND (toLower(e.name) = mention
       OR (size(mention) >= $min_contains AND toLower(e.name) CONTAINS mention))
RETURN DISTINCT e.entity_id AS entity_id, e.name AS name,
       e.type AS type, e.department AS department, mention AS matched,
       CASE WHEN toLower(e.name) = mention THEN 0 ELSE 1 END AS rank
ORDER BY rank, size(e.name)
LIMIT $anchor_limit
"""

# Substring matching below this length produces anchors by coincidence: "approve"
# lands inside "service accounts approved by IT Security", which shares nothing
# with the question but the stem. Exact matches are still accepted at any length,
# so short real names like "PTO" and "VPN" are unaffected.
_MIN_CONTAINS_LENGTH = 5

_DIRECT_CYPHER = """
UNWIND $entity_ids AS entity_id
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {entity_id: entity_id})
MATCH (d:Document {doc_id: c.doc_id})
WHERE d.department IN $departments
RETURN c.chunk_id AS chunk_id, e.name AS entity, e.entity_id AS entity_id,
       1 AS hops
LIMIT $chunk_limit
"""

# One relation hop out. The relationship type is left unconstrained because the
# vocabulary is closed and validated at write time -- an off-ontology predicate
# cannot be in the graph to be matched.
_ONE_HOP_CYPHER = """
UNWIND $entity_ids AS entity_id
MATCH (anchor:Entity {entity_id: entity_id})-[r]-(neighbour:Entity)
WHERE neighbour.department IN $departments
MATCH (c:Chunk)-[:MENTIONS]->(neighbour)
MATCH (d:Document {doc_id: c.doc_id})
WHERE d.department IN $departments
RETURN c.chunk_id AS chunk_id, neighbour.name AS entity,
       neighbour.entity_id AS entity_id, type(r) AS predicate, 2 AS hops
LIMIT $chunk_limit
"""

# A question about a superseded document usually wants the document that
# replaced it, and no amount of text similarity will surface that -- the
# successor may share no wording with the query at all.
_SUPERSEDES_CYPHER = """
UNWIND $doc_ids AS doc_id
MATCH (old:Document {doc_id: doc_id})<-[:SUPERSEDES]-(new:Document)
WHERE new.department IN $departments
MATCH (new)-[:CONTAINS]->(:Section)-[:CONTAINS]->(c:Chunk)
RETURN c.chunk_id AS chunk_id, new.doc_id AS successor_of, 1 AS hops
LIMIT $chunk_limit
"""


def _discriminating(anchors: list[dict]) -> list[dict]:
    """Keep anchors the question really points at; drop coincidental ones.

    "quarterly submarine maintenance schedule" has nothing to do with this
    corpus, but the single word "schedule" sits inside "Discount Schedule" and
    anchored on it. A coincidental hit is not fatal -- fusion and reranking
    demote it -- but at corpus scale one generic word can anchor on hundreds of
    entities, and the graph should not be producing that noise.

    An anchor survives when any of three things is true:

    - the entity's name **is** the mention (an exact match needs no defending);
    - the mention was a **multi-word phrase**, which does not collide by accident;
    - **two or more distinct words** from the question landed on the same
      entity. "tuition" and "reimbursement" both hitting `Tuition
      Reimbursement` is agreement between independent tokens; "schedule" alone
      hitting `Discount Schedule` is a coincidence.

    That third rule is what makes lowercase multi-word questions work, where no
    capitalised phrase exists to extract and no single token matches exactly.
    """
    by_entity: dict[str, set[str]] = {}
    for anchor in anchors:
        by_entity.setdefault(anchor["entity_id"], set()).add(anchor.get("matched") or "")

    kept: list[dict] = []
    seen: set[str] = set()
    for anchor in anchors:
        entity_id = anchor["entity_id"]
        if entity_id in seen:
            continue
        matched = anchor.get("matched") or ""
        if (
            anchor.get("rank") == 0
            or " " in matched
            or len(by_entity[entity_id]) >= 2
        ):
            seen.add(entity_id)
            # How much of the question points at this entity. It is the same
            # agreement signal used to admit the anchor, kept as a weight so
            # scoring can prefer the specific anchor over the generic one: the
            # corpus holds both a bare `reimbursement` entity and a `Tuition
            # Reimbursement` one, and "tuition reimbursement" points two words
            # at the second and one at the first. Unweighted, a mention of
            # either scored the same and the non-reimbursable clauses of the
            # travel policy tied with the section that answers the question.
            kept.append({**anchor, "weight": len(by_entity[entity_id])})
    return kept


class GraphRetriever:
    """Retrieves chunk ids from Neo4j, hydrated through the search index.

    The graph stores chunk *text*, but the retriever deliberately hydrates
    through `HybridSearcher.fetch_chunks` rather than returning the graph's
    copy: that keeps one source of truth for what a chunk says and for its
    version metadata, and it reapplies the department check on the way out. A
    chunk id arriving from the graph is not authorisation to read it.
    """

    def __init__(self, searcher, driver=None) -> None:
        from neo4j import AsyncGraphDatabase

        settings = get_settings()
        self._searcher = searcher
        self._database = settings.neo4j_database
        self._driver = driver or AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    async def ping(self) -> bool:
        async with self._driver.session(database=self._database) as session:
            result = await session.run("RETURN 1 AS ok")
            record = await result.single()
            return bool(record and record["ok"] == 1)

    async def find_anchors(
        self, query: str, *, departments: list[str], limit: int = 40
    ) -> list[dict]:
        mentions = candidate_mentions(query)
        if not mentions or not departments:
            return []
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                _ANCHOR_CYPHER,
                mentions=mentions, departments=departments, anchor_limit=limit,
                min_contains=_MIN_CONTAINS_LENGTH,
            )
            return [dict(record) async for record in result]

    async def retrieve(
        self,
        query: str,
        *,
        departments: list[str],
        top: int | None = None,
    ) -> list[RetrievedChunk]:
        """Chunks reachable from the entities this query names.

        Returns an empty list rather than raising when the graph has no entity
        layer yet, or when nothing in the query names a known entity — the
        graph is one retriever among several, and having nothing to contribute
        is a normal outcome, not a failure.
        """
        settings = get_settings()
        top = top or settings.rerank_top_k
        if not departments:
            return []

        anchors = _discriminating(await self.find_anchors(query, departments=departments))
        if not anchors:
            return []

        entity_ids = [a["entity_id"] for a in anchors]
        anchor_names = {a["entity_id"]: a["name"] for a in anchors}
        anchor_weight = {a["entity_id"]: a.get("weight", 1) for a in anchors}
        chunk_limit = max(top * 3, 30)

        scored: dict[str, tuple[float, str]] = {}

        async with self._driver.session(database=self._database) as session:
            direct = await session.run(
                _DIRECT_CYPHER, entity_ids=entity_ids,
                departments=departments, chunk_limit=chunk_limit,
            )
            async for record in direct:
                chunk_id = record["chunk_id"]
                # A chunk mentioning two of the query's entities is a better
                # answer than one mentioning either alone, so scores accumulate.
                score, path = scored.get(chunk_id, (0.0, ""))
                name = record["entity"]
                credit = _DIRECT_SCORE * anchor_weight.get(record["entity_id"], 1)
                scored[chunk_id] = (
                    score + credit,
                    f"{path}; mentions {name}" if path else f"mentions {name}",
                )

            one_hop = await session.run(
                _ONE_HOP_CYPHER, entity_ids=entity_ids,
                departments=departments, chunk_limit=chunk_limit,
            )
            hops_counted: dict[str, int] = {}
            async for record in one_hop:
                chunk_id = record["chunk_id"]
                # Capped. A section listing five exclusions has five one-hop
                # neighbours and would otherwise accumulate more score than the
                # section that directly answers the question -- rewarding
                # verbosity rather than relevance.
                if hops_counted.get(chunk_id, 0) >= _MAX_ONE_HOP_CREDITS:
                    continue
                hops_counted[chunk_id] = hops_counted.get(chunk_id, 0) + 1
                score, path = scored.get(chunk_id, (0.0, ""))
                step = f"{record['predicate']} → {record['entity']}"
                scored[chunk_id] = (
                    score + _ONE_HOP_SCORE,
                    f"{path}; {step}" if path else step,
                )

        if not scored:
            return []

        ranked = sorted(scored.items(), key=lambda kv: kv[1][0], reverse=True)[:top]
        chunk_ids = [chunk_id for chunk_id, _ in ranked]
        hydrated = await self._searcher.fetch_chunks(
            chunk_ids, departments=departments
        )

        by_id = {c.chunk_id: c for c in hydrated}
        out: list[RetrievedChunk] = []
        for chunk_id, (score, path) in ranked:
            chunk = by_id.get(chunk_id)
            if chunk is None:
                # In the graph but not in the index: the two stores have
                # drifted. Worth knowing about, not worth failing a query over.
                logger.debug("graph chunk %s not present in the search index", chunk_id)
                continue
            chunk.score = score
            chunk.retrievers = ["graph"]
            chunk.matched_queries = [query]
            chunk.graph_path = path
            out.append(chunk)

        logger.debug(
            "graph retrieval: %d anchors (%s) → %d chunks",
            len(anchors), ", ".join(sorted(set(anchor_names.values())))[:120], len(out),
        )
        return out

    async def successors_of(
        self, doc_ids: list[str], *, departments: list[str], limit: int = 10
    ) -> list[RetrievedChunk]:
        """Chunks from documents that supersede the given ones.

        Text similarity cannot find these: the replacement document may share
        no wording with the question that surfaced the original.
        """
        if not doc_ids or not departments:
            return []
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                _SUPERSEDES_CYPHER, doc_ids=doc_ids,
                departments=departments, chunk_limit=limit,
            )
            chunk_ids = [record["chunk_id"] async for record in result]

        hydrated = await self._searcher.fetch_chunks(
            chunk_ids[:limit], departments=departments
        )
        for chunk in hydrated:
            chunk.score = _SUPERSEDES_SCORE
            chunk.retrievers = ["graph:supersedes"]
            chunk.graph_path = "supersedes a retrieved document"
        return hydrated

    async def aclose(self) -> None:
        await self._driver.close()
