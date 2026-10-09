# app/researcher_adapter.py
import asyncio
import re
import logging
import time
import hashlib
from pathlib import Path
from urllib.parse import urlparse, urlsplit, urlunsplit
from typing import Dict, Any
from gpt_researcher import GPTResearcher

from app.progress_adapter import ProgressReporter, GPTResearcherCallbackHandler
from app.model_adapter import RequestEnvironmentManager
from app.security import PROFILE_DOMAINS, enforce_egress_protection, is_safe_url, is_provider_host, search_budget_exhausted
from app.config import brain_memory_client, settings
import app.storage as storage_module


class _LeaseLostError(RuntimeError):
    """Raised internally when this worker no longer owns an operation."""

logger = logging.getLogger("web-intelligence")
import psutil
import os

ESTIMATED_INPUT_TOKEN_RATE_USD = 0.0000025
ESTIMATED_OUTPUT_TOKEN_RATE_USD = 0.000010
MAX_INPUT_CONTEXT_BYTES = 120_000
MAX_INPUT_FILE_BYTES = 40_000
MAX_INPUT_CHUNKS = 12
MAX_REPOSITORY_PATHS_VISITED = 500
SUPPORTED_INPUT_EXTENSIONS = {
    ".md", ".markdown", ".txt", ".rst", ".py", ".js", ".jsx", ".ts", ".tsx",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".css", ".html", ".sql"
}
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "with"
}
NEGATION_TERMS = {"no", "not", "never", "none", "without", "cannot", "can't", "isn't", "wasn't", "won't"}
# Claims are only emitted for report sentences with enough substance to be
# meaningful. This is what keeps the report-derived fallback from minting a
# claim+evidence pair for every trivial fragment.
MIN_CLAIM_SENTENCE_CHARS = 24

def get_memory_usage_mb() -> float:
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()) * 4 // 3)

def estimate_model_calls(mode: str) -> int:
    return 4 if mode == "deep" else 2

def estimate_model_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * ESTIMATED_INPUT_TOKEN_RATE_USD) + (output_tokens * ESTIMATED_OUTPUT_TOKEN_RATE_USD)

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def trim_to_token_budget(text: str, maximum_tokens: int) -> str:
    words = text.split()
    maximum_words = max(1, maximum_tokens * 3 // 4)
    if len(words) <= maximum_words:
        return text
    return " ".join(words[:maximum_words]) + "\n\n[Truncated to satisfy maximumModelTokens.]"

def split_sentences(text: str) -> list[str]:
    candidates = re.split(r"(?<=[.!?])\s+", text.replace("\n", " ").strip())
    return [candidate.strip() for candidate in candidates if candidate.strip()]


def select_claim_sentences(text: str, maximum: int) -> list[str]:
    """Substantive report sentences eligible to become claims/evidence."""
    return [
        sentence for sentence in split_sentences(text)
        if len(sentence) >= MIN_CLAIM_SENTENCE_CHARS
    ][:maximum]

def normalize_source_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(normalize_source_text(item) for item in value)
    if isinstance(value, dict):
        for key in ("content", "raw_content", "summary", "text", "body"):
            if value.get(key):
                return normalize_source_text(value[key])
    return str(value)

def normalized_tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9][a-z0-9-]{2,}", text.lower())
        if token not in STOP_WORDS
    }

def negation_present(text: str) -> bool:
    tokens = set(re.findall(r"[a-z']+", text.lower()))
    return bool(tokens & NEGATION_TERMS)

def passage_support_score(claim: str, passage: str) -> float:
    claim_tokens = normalized_tokens(claim)
    if not claim_tokens:
        return 0.0
    passage_tokens = normalized_tokens(passage)
    if not passage_tokens:
        return 0.0
    return len(claim_tokens & passage_tokens) / len(claim_tokens)

def verification_status_for_score(claim: str, passage: str, score: float) -> str:
    if score < 0.35:
        return "unsupported"
    if negation_present(claim) != negation_present(passage) and score >= 0.55:
        return "conflicting"
    if score >= 0.72:
        return "supported"
    return "partially-supported"

def source_url_from_record(record: dict) -> str:
    for key in ("url", "source", "link", "href"):
        if record.get(key):
            return str(record[key])
    return ""

def source_title_from_record(record: dict, fallback: str) -> str:
    for key in ("title", "name", "source"):
        if record.get(key):
            return str(record[key])
    return fallback

def source_metadata_from_record(record: dict) -> dict:
    metadata = {}
    for output_key, candidates in {
        "title": ("title", "name", "source"),
        "publisher": ("publisher", "site_name", "domain", "source_name"),
        "author": ("author", "byline"),
        "publishedAt": ("publishedAt", "published_at", "published_date", "date"),
    }.items():
        for candidate in candidates:
            if record.get(candidate):
                metadata[output_key] = str(record[candidate])
                break

    score = record.get("qualityScore", record.get("quality_score", record.get("score")))
    if score is not None:
        try:
            metadata["qualityScore"] = max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            pass
    return metadata

def publisher_from_url(url: str) -> str | None:
    host = urlparse(url).hostname
    return host.lower() if host else None

def collect_source_metadata(researcher, search_results: list[dict]) -> dict[str, dict]:
    metadata_by_url = {}
    records = []
    try:
        records.extend(raw for raw in researcher.get_research_sources() or [] if isinstance(raw, dict))
    except Exception:
        logger.debug("Unable to read research source metadata", exc_info=True)
    records.extend(raw for raw in search_results if isinstance(raw, dict))

    for record in records:
        url = source_url_from_record(record)
        if not url:
            continue
        metadata_by_url.setdefault(url, {}).update(source_metadata_from_record(record))
    return metadata_by_url

def collect_passage_records(researcher, safe_source_urls: list[str]) -> list[dict]:
    source_url_set = set(safe_source_urls)
    records = []

    for raw in researcher.get_research_sources() or []:
        if not isinstance(raw, dict):
            continue
        url = source_url_from_record(raw)
        if url and url not in source_url_set:
            continue
        text = normalize_source_text(raw)
        if text:
            records.append({
                "url": url,
                "title": source_title_from_record(raw, "Research source"),
                "text": text
            })

    context_items = researcher.get_research_context() or []
    if isinstance(context_items, str):
        context_items = [context_items]
    for item in context_items:
        text = normalize_source_text(item)
        if not text:
            continue
        url = ""
        for candidate in safe_source_urls:
            if candidate in text:
                url = candidate
                break
        records.append({
            "url": url,
            "title": "Research context",
            "text": text
        })

    return records

def select_passages(text: str, maximum_passages: int = 3) -> list[str]:
    sentences = [sentence for sentence in split_sentences(text) if len(sentence) >= MIN_CLAIM_SENTENCE_CHARS]
    if sentences:
        return sentences[:maximum_passages]
    compact = " ".join(text.split())
    return [compact[:500]] if compact else []

def build_structured_findings_from_passages(op_id: str, passage_records: list[dict], sources: list[dict], maximum_items: int = 12) -> tuple[list[dict], list[dict], list[dict]]:
    evidence = []
    claims = []
    source_by_url = {source["url"]: source for source in sources}
    default_source = sources[0] if sources else None
    citations_by_source = {
        source["id"]: {
            "id": f"cite-{op_id}-{idx}",
            "sourceId": source["id"],
            "evidenceIds": [],
            "claimIds": []
        }
        for idx, source in enumerate(sources)
    }

    seen_passages = set()
    for record in passage_records:
        if len(evidence) >= maximum_items:
            break
        source = source_by_url.get(record.get("url"))
        if source is None:
            # A passage whose URL is absent or not a known source cannot be
            # attributed. Only adopt the single-source fallback when there is
            # exactly one candidate, so a passage is never pinned to an
            # unrelated source it cannot be traced to.
            if len(sources) == 1 and not record.get("url"):
                source = default_source
            else:
                continue
        for passage in select_passages(record.get("text", "")):
            if len(evidence) >= maximum_items:
                break
            if passage in seen_passages:
                continue
            seen_passages.add(passage)
            idx = len(evidence)
            evidence_id = f"ev-{op_id}-{idx}"
            evidence.append({
                "id": evidence_id,
                "sourceId": source["id"],
                "section": record.get("title"),
                "passage": passage,
                "contentHash": content_hash(passage),
                "relevanceScore": 0.9
            })
            citations_by_source[source["id"]]["evidenceIds"].append(evidence_id)

    return evidence, claims, list(citations_by_source.values())

def verify_claims_against_evidence(op_id: str, report_text: str, evidence: list[dict], citations: list[dict], maximum_claims: int = 10) -> list[dict]:
    claims = []
    citation_by_evidence = {}
    for citation in citations:
        for evidence_id in citation.get("evidenceIds", []):
            citation_by_evidence[evidence_id] = citation

    for claim_text in select_claim_sentences(report_text, maximum_claims):
        best_evidence = None
        best_score = 0.0
        for item in evidence:
            score = passage_support_score(claim_text, item.get("passage", ""))
            if score > best_score:
                best_score = score
                best_evidence = item

        claim_id = f"claim-{op_id}-{len(claims)}"
        evidence_ids = []
        confidence = 0.35
        status = "unsupported"
        if best_evidence:
            status = verification_status_for_score(claim_text, best_evidence.get("passage", ""), best_score)
            if status != "unsupported":
                evidence_ids = [best_evidence["id"]]
                confidence = max(0.45, min(0.95, best_score))
                citation = citation_by_evidence.get(best_evidence["id"])
                if citation and claim_id not in citation["claimIds"]:
                    citation["claimIds"].append(claim_id)

        claims.append({
            "id": claim_id,
            "text": claim_text,
            "evidenceIds": evidence_ids,
            "confidence": confidence,
            "verificationStatus": status
        })

    return claims

def build_structured_findings(op_id: str, report_text: str, sources: list[dict], maximum_items: int = 8) -> tuple[list[dict], list[dict], list[dict]]:
    """Report-derived fallback used only when no source passage text is available.

    A report sentence cannot be attributed to a specific source without passage
    text, so no evidence is fabricated here. Claims are emitted as ``inferred``
    with no evidence IDs and citations are left empty. Presenting index-ordered
    attribution as supporting evidence would overstate provenance for a service
    whose contract is source-backed evidence.
    """
    citations = [
        {
            "id": f"cite-{op_id}-{idx}",
            "sourceId": source["id"],
            "evidenceIds": [],
            "claimIds": []
        }
        for idx, source in enumerate(sources)
    ]

    claims = []
    for sentence in select_claim_sentences(report_text, maximum_items):
        claims.append({
            "id": f"claim-{op_id}-{len(claims)}",
            "text": sentence,
            "evidenceIds": [],
            "confidence": 0.4,
            "verificationStatus": "inferred"
        })

    return [], claims, citations

def freshness_instruction(freshness: Dict[str, str] | None) -> str:
    if not freshness:
        return ""
    parts = []
    if freshness.get("since"):
        parts.append(f"prefer sources published on or after {freshness['since']}")
    if freshness.get("until"):
        parts.append(f"exclude sources published after {freshness['until']}")
    if freshness.get("maxAgeDays"):
        parts.append(f"prefer sources from the last {freshness['maxAgeDays']} days")
    return "; ".join(parts)

def allowed_input_roots() -> list[Path]:
    """Configured roots that local document/repository inputs may be read from.

    Local input ingestion reads caller-supplied filesystem paths, so it must be
    confined to explicitly authorized roots rather than any absolute path.
    """
    roots = []
    for raw in (settings.LOCAL_INPUT_ROOTS or "").split(os.pathsep):
        raw = raw.strip()
        if not raw:
            continue
        roots.append(Path(raw).expanduser().resolve())
    return roots

def is_within_allowed_roots(path: Path) -> bool:
    """True when path, once canonicalized, lies within a configured root.

    This is a boolean guard, not a value-returning "give me the safe path"
    helper: callers must check ``if not is_within_allowed_roots(path): return``
    and then keep using that *same* ``path`` object (optionally transformed by
    a non-filesystem-touching call such as ``.expanduser()``) for every
    subsequent filesystem operation, rather than substituting a value this
    function computed internally.

    That distinction is what CodeQL's uncontrolled-path-expression sanitizer
    recognizes: a guard on the same variable that later reaches a filesystem
    sink clears it, but a helper that instead *returns* a freshly resolved
    value is -- from a static data-flow point of view -- still returning data
    derived from the same caller-supplied path, so a filesystem call on that
    returned value downstream is flagged as unsanitized even though this
    function already validated it. The real canonicalization
    (``os.path.realpath``, which also resolves symlinks/``..``/``~``) still
    happens here so an equivalent spelling of an allowed root is accepted and
    a symlink escape is rejected; it is used only for the containment
    decision, never returned.
    """
    resolved_str = os.path.realpath(os.path.expanduser(str(path)))
    return _real_path_within_allowed_roots(resolved_str)

def _real_path_within_allowed_roots(resolved_str: str) -> bool:
    """Containment check on an already-canonicalized path string."""
    for root in allowed_input_roots():
        root_str = str(root)
        # root_str already ends with os.sep when it is the filesystem root
        # itself (e.g. "/"); appending another separator there would produce
        # "//" and reject every child path, so only add one when it is not
        # already present.
        prefix = root_str if root_str.endswith(os.sep) else root_str + os.sep
        if resolved_str == root_str or resolved_str.startswith(prefix):
            return True
    return False

def _open_validated_file(path: Path):
    """Open path for reading, closing the check-to-open symlink-retarget race.

    is_within_allowed_roots() validates path's canonical target at check time,
    but if path (or a directory in it) is a symlink another local process
    could retarget it to point outside LOCAL_INPUT_ROOTS between that check
    and a later open() call. Opening first and then verifying the path of the
    *actual opened file descriptor* (via ``/dev/fd``, which reflects whatever
    the open() syscall itself resolved to, atomically) closes that window:
    there is no second, separately racy resolve-then-open step.
    """
    fd = os.open(str(path), os.O_RDONLY)
    try:
        real = os.path.realpath(f"/dev/fd/{fd}")
        if not _real_path_within_allowed_roots(real):
            raise PermissionError(
                "Opened file resolved outside LOCAL_INPUT_ROOTS after the containment check."
            )
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise

def input_text_from_file(path: Path) -> str:
    if settings.DEPLOYMENT_MODE != "local":
        return ""
    if not is_within_allowed_roots(path):
        # Caller-supplied absolute paths can disclose sensitive filenames and
        # carry newline/control characters, so log only a bounded digest.
        logger.warning(
            "Refusing local input path outside LOCAL_INPUT_ROOTS (path hash %s)",
            hashlib.sha256(str(path).encode()).hexdigest()[:12],
        )
        return ""
    path = path.expanduser()
    if not path.is_file() or path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
        return ""
    try:
        with _open_validated_file(path) as input_file:
            raw = input_file.read(MAX_INPUT_FILE_BYTES)
        return raw.decode("utf-8", errors="replace")
    except (OSError, PermissionError):
        # Do not log the caller-supplied path: it can disclose sensitive
        # filenames and carry newline/control characters into the log.
        logger.warning("Unable to read a declared local input file (suffix %s).", path.suffix)
        return ""

def collect_document_context(documents: list[dict], remaining_chunks: int = MAX_INPUT_CHUNKS) -> list[dict]:
    chunks = []
    for item in documents:
        if len(chunks) >= remaining_chunks:
            break
        if not isinstance(item, dict) or not item.get("path"):
            continue
        path = Path(str(item["path"])).expanduser()
        text = input_text_from_file(path)
        if text:
            chunks.append({
                "label": item.get("displayName") or path.name,
                "path": str(path),
                "text": text
            })
    return chunks

def collect_repository_context(repositories: list[dict], remaining_chunks: int = MAX_INPUT_CHUNKS) -> list[dict]:
    chunks = []
    for item in repositories:
        if len(chunks) >= remaining_chunks:
            break
        if not isinstance(item, dict) or not item.get("path"):
            continue
        # is_within_allowed_roots() canonicalizes internally (so an equivalent
        # spelling -- trailing slash, ~, .., symlink -- of an allowed
        # directory is still usable, and a symlink escape is still rejected)
        # but is checked here as a boolean guard on `root` itself: every
        # filesystem operation below uses this same validated `root` object,
        # never a value resolve_within_allowed_roots computed separately. See
        # is_within_allowed_roots()'s docstring for why that distinction
        # matters for the uncontrolled-path-expression sanitizer.
        root = Path(str(item["path"])).expanduser()
        if not is_within_allowed_roots(root) or not root.is_dir():
            logger.warning(
                "Refusing repository input outside LOCAL_INPUT_ROOTS or not a directory (path hash %s)",
                hashlib.sha256(str(item["path"]).encode()).hexdigest()[:12],
            )
            continue
        visited = 0
        for current_root, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                dirname for dirname in dirnames
                if dirname not in {".git", "node_modules", ".venv", "__pycache__", "dist", "build"}
            ]
            for filename in sorted(filenames):
                if len(chunks) >= remaining_chunks or visited >= MAX_REPOSITORY_PATHS_VISITED:
                    break
                visited += 1
                path = Path(current_root) / filename
                if not path.is_file() or path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
                    continue
                text = input_text_from_file(path)
                if text:
                    chunks.append({
                        "label": str(path.relative_to(root)),
                        "path": str(path),
                        "repository": root.name,
                        "branch": item.get("branch"),
                        "text": text
                    })
            if len(chunks) >= remaining_chunks or visited >= MAX_REPOSITORY_PATHS_VISITED:
                break
    return chunks

def collect_input_context(inputs: Dict[str, Any] | None) -> tuple[list[dict], bool]:
    if not isinstance(inputs, dict):
        return [], False
    if settings.DEPLOYMENT_MODE != "local":
        return [], False
    chunks = []
    documents = inputs.get("documents") or []
    repositories = inputs.get("repositories") or []
    if isinstance(documents, list):
        chunks.extend(collect_document_context(documents, MAX_INPUT_CHUNKS))
    if isinstance(repositories, list) and len(chunks) < MAX_INPUT_CHUNKS:
        chunks.extend(collect_repository_context(repositories, MAX_INPUT_CHUNKS - len(chunks)))

    total_bytes = 0
    bounded_chunks = []
    for chunk in chunks:
        encoded = chunk["text"].encode("utf-8", errors="ignore")
        remaining = MAX_INPUT_CONTEXT_BYTES - total_bytes
        if remaining <= 0:
            break
        if len(encoded) > remaining:
            chunk = dict(chunk)
            chunk["text"] = encoded[:remaining].decode("utf-8", errors="ignore")
        total_bytes += len(chunk["text"].encode("utf-8", errors="ignore"))
        bounded_chunks.append(chunk)
    return bounded_chunks, inputs.get("allowExternalUse") is True

def _has_unread_declared_inputs(raw_inputs: Dict[str, Any] | None, input_chunks: list[dict]) -> bool:
    if not isinstance(raw_inputs, dict):
        return False
    documents = raw_inputs.get("documents") or []
    repositories = raw_inputs.get("repositories") or []
    if not isinstance(documents, list) or not isinstance(repositories, list):
        return False
    read_documents = sum(1 for chunk in input_chunks if not chunk.get("repository"))
    read_repositories = {chunk.get("repository") for chunk in input_chunks if chunk.get("repository")}
    return len(documents) > read_documents or len(repositories) > len(read_repositories)

def format_input_context_for_query(chunks: list[dict]) -> str:
    sections = []
    for chunk in chunks:
        sections.append(f"Input: {chunk['label']}\n{chunk['text']}")
    return "\n\n---\n\n".join(sections)

def build_effective_query(
    query: str,
    freshness: Dict[str, str] | None,
    input_chunks: list[dict],
    allow_external_inputs: bool,
    raw_inputs: Dict[str, Any] | None = None
) -> tuple[str, list[str]]:
    additions = []
    limitations = []
    fresh = freshness_instruction(freshness)
    if fresh:
        additions.append(f"Freshness constraint: {fresh}.")
    if input_chunks:
        limitations.append("Local document/repository inputs were processed as bounded first-party evidence.")
        if _has_unread_declared_inputs(raw_inputs, input_chunks):
            # Some declared inputs produced no readable chunk (unreadable path,
            # unsupported extension, outside LOCAL_INPUT_ROOTS). Say so instead
            # of implying every declared input was honored.
            limitations.append("Some declared local inputs were not readable and were skipped.")
        if allow_external_inputs:
            additions.append("Use this explicitly provided local input context as first-party context:\n" + format_input_context_for_query(input_chunks))
            limitations.append("Local inputs were explicitly allowed for external research prompt context.")
        else:
            limitations.append("Local inputs were not sent to external research providers because inputs.allowExternalUse was not true.")
    elif raw_inputs and (raw_inputs.get("documents") or raw_inputs.get("repositories")):
        # Inputs were requested but produced no usable chunks (for example a
        # path outside LOCAL_INPUT_ROOTS). Still report that they were declared,
        # so the caller is not misled into thinking inputs were honored.
        limitations.append("Local document/repository inputs were declared but no readable content was available.")
        if raw_inputs.get("allowExternalUse") is not True:
            limitations.append("Local inputs were not sent to external research providers because inputs.allowExternalUse was not true.")
    if not additions:
        return query, limitations
    return query + "\n\n" + "\n\n".join(additions), limitations

def append_input_sources(op_id: str, input_chunks: list[dict], sources: list[dict], evidence: list[dict], citations: list[dict]) -> None:
    for chunk in input_chunks:
        source_id = f"src-{op_id}-{len(sources)}"
        source_type = "repository" if chunk.get("repository") else "document"
        source = {
            "id": source_id,
            # url is the HTTP(S) contract field; a local file has no HTTP URL,
            # so the locator is carried on uri and url stays empty rather than
            # smuggling a file:// pseudo-URL through an HTTP-shaped field.
            "url": "",
            # Path.as_uri() percent-encodes reserved characters and rejects
            # relative paths, so the locator is a syntactically valid URI.
            "uri": Path(chunk["path"]).resolve().as_uri(),
            "title": chunk["label"],
            "retrievedAt": int(time.time() * 1000),
            "sourceType": source_type,
            "qualityScore": 1.0
        }
        sources.append(source)
        citation = {
            "id": f"cite-{op_id}-{len(citations)}",
            "sourceId": source_id,
            "evidenceIds": [],
            "claimIds": []
        }
        for passage in select_passages(chunk["text"], maximum_passages=2):
            evidence_id = f"ev-{op_id}-{len(evidence)}"
            evidence.append({
                "id": evidence_id,
                "sourceId": source_id,
                "section": chunk["label"],
                "passage": passage,
                "contentHash": content_hash(passage),
                "relevanceScore": 1.0
            })
            citation["evidenceIds"].append(evidence_id)
        citations.append(citation)

_pending_ingest_tasks: set = set()


def _public_locator(raw: str) -> str:
    """Reduce a source locator to something safe to persist in shared memory.

    Reduces a URL to its origin. A secret can ride in userinfo, in the query, or
    in the path itself (a magic-link token, "/reset/<token>", ";jsessionid="),
    and a local input's locator is an absolute filesystem path the operator may
    never have cleared for external use (allowExternalUse unset or false).
    is_safe_url() validates only scheme, host and resolved address, so none of
    that is filtered upstream. Brain persists what it is handed and replays it
    through recall_context() into later model prompts, so the redaction has to
    happen before ingestion — there is no read-side filter to fall back on.

    The cost is deliberate: provenance drops to publisher level, and two claims
    from one site become indistinguishable by locator. The authoritative
    research result keeps the full URL; only this shared-memory copy fails
    closed.
    """
    try:
        parts = urlsplit(raw)
        if parts.scheme not in ("http", "https"):
            # file:// and anything else: omit rather than publish a local path.
            return ""
        host = parts.hostname or ""
        if not host:
            return ""
        if parts.query:
            # The query can carry the secret (signature, token) or the resource
            # identity (/article?id=123) and there is no general way to tell
            # which. Stripping it would silently publish a locator for a
            # different page, so omit it entirely: no provenance beats wrong
            # provenance.
            return ""
        # .port parses lazily and raises on a malformed value like ":bad",
        # which is_safe_url() does not inspect. It stays inside the guard
        # because this runs while building the ingest task arguments: an
        # escaping exception would fail research that already succeeded and
        # was already saved, for the sake of an optional memory record.
        port = parts.port
    except ValueError:
        return ""
    # hostname strips the brackets an IPv6 literal needs, so put them back —
    # otherwise the rebuilt locator cannot identify the evidence host.
    netloc = f"[{host}]" if ":" in host else host
    if port:
        netloc = f"{netloc}:{port}"
    # Origin only. The path is dropped for the same reason the query is: it can
    # carry a capability credential — a magic-link token, /reset/<token>, a
    # ";jsessionid=" path parameter — and there is no general way to tell one
    # from an identifying segment like /article/12345. is_safe_url() validates
    # the destination, never the path's contents. The authoritative research
    # result still holds the complete URL; this is the shared-memory copy, so
    # it fails closed to publisher-level provenance.
    return urlunsplit((parts.scheme, netloc, "", "", ""))


_URL_IN_CLAIM_TEXT = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S+")

# Scheme-relative form ("//example.com/reset/<token>"). Matched separately and
# more strictly than the scheme-ful pattern: bare "//" also opens a line comment
# in most languages this agent researches, so the authority must look like a
# host — a dotted TLD or an explicit port — or "// see the notes below" would
# cost a legitimate finding its retention. The lookbehind keeps it from
# re-matching the "//" inside a scheme-ful URL.
# A TLD label, punycode included: "com", "museum", "xn--p1ai". Hyphens are
# allowed inside but never at the end. Defined once and shared by every matcher:
# the last two rounds were each the same gap found in a different branch, so a
# single definition is what keeps them from drifting apart again.
# Unicode-aware: "com", "xn--p1ai" and "\u0440\u0444" are all valid TLD labels. [^\W\d_] is a
# letter in any script, [^\W_] a letter or digit — so a label may contain
# hyphens but never start or end with one.
_TLD = r"[^\W\d_][\w-]*[^\W_]"
# A host label, same alphabet.
_HOST = r"[\w.-]"


_SCHEME_RELATIVE_URL_IN_CLAIM_TEXT = re.compile(
    r"(?<![A-Za-z0-9:])//"
    r"(?:[^\s/?#@]*@)?"
    # A bracketed IPv6 literal is as unambiguous an authority as a dotted host,
    # and _public_locator() already round-trips one — the two were simply
    # inconsistent.
    r"(?:\[[0-9A-Fa-f:.]+\](?::\d{1,5})?"
    # An unbracketed IPv4 literal has no alphabetic TLD and needs no port, so
    # the host alternative below never matched it. Safe to accept here because
    # the "//" prefix is what disambiguates: a bare "1.2.3.4" in prose is a
    # version string, "//1.2.3.4" is an authority.
    r"|\d{1,3}(?:\.\d{1,3}){3}(?::\d{1,5})?"
    r"|" + _HOST + r"*(?:\." + _TLD + r"|:\d{1,5}))"
    # ":" and ";" start a continuation too — a port, or a ";jsessionid=" path
    # parameter. Without them the match stops at the bare authority, which then
    # reduces to a safe origin and lets the credential-bearing tail through.
    # A sentence colon ("at //example.test: it is fast") is handled by
    # _trim_sentence_punctuation, which strips the trailing ":" back off.
    r"(?:[:/?#;]\S*)?"
)

# Schemeless host-shaped links, limited to the two unambiguous shapes: a "www."
# prefix, or a dotted host carrying a query string.
#
# A general "host.tld/path" rule is deliberately NOT used. "github.com/owner/repo"
# is a Go module path, "docs.python.org/3/library/urllib.html" is an ordinary
# citation, and this agent researches exactly that kind of text — the broad form
# would withhold legitimate findings far more often than it caught a capability
# link, and silently shrinking retention is its own failure. The residue that
# leaves (a schemeless token link with neither "www." nor a query) is real and
# is not covered here.
_SCHEMELESS_URL_IN_CLAIM_TEXT = re.compile(
    r"(?<![/@\w.])(?:"
    r"www\." + _HOST + r"+\." + _TLD + r"(?:[:/?#;]\S*)?"
    r"|" + _HOST + r"+\." + _TLD + r"(?:/\S*)?\?\S+"
    r")"
)

# Well-known credential formats, matched by their issuer-assigned prefix and
# length. Deliberately prefix-anchored rather than entropy-based: this is a
# technical research agent, so commit SHAs, UUIDs, digests and base64 payloads
# are ordinary subject matter, and a generic high-entropy rule would withhold
# legitimate findings far more often than it caught a secret.
_CREDENTIAL_PATTERNS = (
    re.compile(r"\b[sp]k-[A-Za-z0-9_-]{16,}"),                                      # OpenAI-style
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),                                    # GitHub token
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),                                  # GitHub fine-grained PAT
    re.compile(r"\bglpat-[A-Za-z0-9_-]{16,}"),                                      # GitLab PAT
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),                                  # Slack
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),                                   # AWS access key id
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"),                                        # Google API key
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),    # JWT
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),                              # PEM private key
)


_CLOSERS = {")": "(", "]": "[", "}": "{"}


def _trim_sentence_punctuation(candidate: str) -> str:
    """Strip trailing sentence punctuation from a URL match.

    A closing bracket is removed only when it is unbalanced, so "(see
    https://example.test/a)" loses its ")" while "//[2606:4700:4700::1111]"
    keeps the "]" that closes its IPv6 literal. A blanket rstrip ate that
    bracket and turned a valid bare origin into an unparseable string, which
    then failed closed and withheld the claim.
    """
    while candidate:
        last = candidate[-1]
        if last in ".,;:!?\"'":
            candidate = candidate[:-1]
        elif last in _CLOSERS and candidate.count(last) > candidate.count(_CLOSERS[last]):
            candidate = candidate[:-1]
        else:
            break
    return candidate


def _claim_text_carries_a_secret(text: str) -> bool:
    """A report sentence can quote a presigned URL, magic link or session id
    copied from an authenticated page.

    _public_locator() sanitizes only the separate source locator, so the claim
    text needs the same policy — applied by *reuse*, so the two cannot drift
    apart as that policy changes. A URL embedded in a claim is cleared only when
    it already equals what _public_locator() would reduce it to: a bare origin.

    Anything carrying userinfo, a query, a fragment or a path makes the whole
    claim ineligible rather than being rewritten in place, because silently
    editing a verified claim is exactly what the truncation rule forbids.

    A bare credential outside any URI ("The API key is sk-proj-...") is caught
    separately, by matching well-known issuer prefixes.

    Known limit, stated rather than papered over: prefix matching cannot be
    complete. A novel, internal, or unprefixed secret still passes, so this
    narrows the exposure — it does not close it. Treat the retained record as
    reduced-risk, never as guaranteed secret-free.
    """
    for match in _URL_IN_CLAIM_TEXT.findall(text):
        candidate = _trim_sentence_punctuation(match)
        if _public_locator(candidate) != candidate:
            return True
    for match in _SCHEME_RELATIVE_URL_IN_CLAIM_TEXT.findall(text):
        # Resolved against a scheme so the same locator policy decides it:
        # "//host" survives as a bare origin, "//host/reset/<token>" does not.
        candidate = "https:" + _trim_sentence_punctuation(match)
        if _public_locator(candidate) != candidate:
            return True
    for match in _SCHEMELESS_URL_IN_CLAIM_TEXT.findall(text):
        candidate = "https://" + _trim_sentence_punctuation(match)
        if _public_locator(candidate) != candidate:
            return True
    return any(pattern.search(text) for pattern in _CREDENTIAL_PATTERNS)


def _retained_findings(verified_claims: list, evidence: list, sources: list, allow_external_inputs: bool = False) -> tuple[list[dict], int, int]:
    """Pair each verified claim with the URL of the source its evidence came from.

    A claim records evidence ids, not a source, so the locator is resolved
    claim -> evidence -> source. A claim whose evidence resolves to no locator is
    still retained: the finding is what makes the record recallable, and dropping
    it for want of a URL would discard verified knowledge.

    Local-source findings are withheld unless inputs.allowExternalUse was true.
    append_input_sources() feeds document and repository passages into claim
    verification regardless of that flag, so a report sentence can be supported
    by a confidential local passage. Brain is shared memory and recall_context()
    can replay it into an external model prompt, so retention needs the same
    consent the external research path required. An unresolved source type
    cannot be shown to be web either, so it fails closed with the rest.

    Returns (findings, withheld_count) — the count is recorded in the artifact
    so the gap between verified claims and retained findings is never silent.
    """
    provenance_by_source = {
        source.get("id"): (
            _public_locator(source.get("url") or source.get("uri") or ""),
            str(source.get("sourceType") or ""),
        )
        for source in sources
    }
    source_by_evidence = {item.get("id"): item.get("sourceId") for item in evidence}
    findings, withheld, secret_bearing = [], 0, 0
    for claim in verified_claims:
        locator, source_type = "", ""
        for evidence_id in claim.get("evidenceIds", []):
            locator, source_type = provenance_by_source.get(source_by_evidence.get(evidence_id), ("", ""))
            # A local document redacts to no locator but still has a type worth
            # carrying, so settle on the first evidence that resolves to either.
            if locator or source_type:
                break
        if not allow_external_inputs and source_type != "web":
            withheld += 1
            continue
        text = claim.get("text", "")
        if _claim_text_carries_a_secret(text):
            secret_bearing += 1
            continue
        findings.append({"text": text, "url": locator, "sourceType": source_type})
    return findings, withheld, secret_bearing


def _schedule_outcome_ingest(client, result: Dict[str, Any], op_id: str, mode: str, allow_external_inputs: bool = False) -> None:
    """Persist an optional Brain outcome best-effort, off the critical path.

    The task is kept referenced so it is not garbage-collected mid-flight and
    tracked in ``_pending_ingest_tasks`` so tests and shutdown can await an
    explicit completion signal.
    """
    # Count only claims carrying independently extracted source evidence.
    # The report-only fallback in build_structured_findings() manufactures
    # evidence from the generated report itself and labels those claims
    # "partially-supported", so counting that status would ingest a fabricated
    # verification success. "supported" against a real passage (or any status
    # backed by an evidence id) is the independently evidenced shape.
    verified_claims = [
        claim for claim in result.get("claims", [])
        if claim.get("verificationStatus") == "supported" and claim.get("evidenceIds")
    ]
    sources = result.get("sources", [])
    findings, withheld, secret_bearing = _retained_findings(
        verified_claims, result.get("evidence", []), sources, allow_external_inputs
    )
    task = asyncio.create_task(asyncio.to_thread(
        client.ingest_verified_outcome,
        operation_id=op_id, status=result["status"], mode=mode,
        # The counters stay whole: they are aggregate and carry no content, so
        # withholding a finding must not also understate what was verified.
        source_count=len(sources), verified_claim_count=len(verified_claims),
        source_types=[source.get("sourceType", "") for source in sources],
        findings=findings, withheld_findings=withheld,
        secret_bearing_findings=secret_bearing,
    ))
    _pending_ingest_tasks.add(task)
    task.add_done_callback(_pending_ingest_tasks.discard)


async def flush_pending_ingest_tasks(timeout: float = 6.0) -> int:
    """Await best-effort ingestion tasks. Tests and shutdown call this."""
    pending = [task for task in _pending_ingest_tasks if not task.done()]
    if not pending:
        return 0
    await asyncio.wait(pending, timeout=timeout)
    return len(pending)


def schedule_outcome_ingest(result: Dict[str, Any], op_id: str, mode: str, inputs: Dict[str, Any] | None = None) -> bool:
    """Schedule the optional Brain outcome ingest once the result is durable.

    Called by the caller after a successful ``storage.save_operation`` so a
    completed/partial research run is never recorded in Brain unless it was
    persisted. Returns True only when a task was scheduled.
    """
    if result.get("status") not in ("completed", "partial"):
        return False
    client = brain_memory_client()
    if not client:
        return False
    # Consent is read from the request's own inputs, not from the result:
    # the result never carried the flag, which is how local passages reached
    # retention unchecked in the first place.
    _schedule_outcome_ingest(client, result, op_id, mode, (inputs or {}).get("allowExternalUse") is True)
    return True


async def conduct_web_research(
    op_id: str,
    query: str,
    mode: str,
    profile: str,
    limits: Dict[str, Any],
    source_policy: Dict[str, Any] | None,
    freshness: Dict[str, str] | None,
    inputs: Dict[str, Any] | None,
    model_provider: str | None,
    model_name: str | None,
    require_claim_verification: bool,
    reporter: ProgressReporter,
    headers: Dict[str, str]
) -> Dict[str, Any]:
    start_time = time.time()

    # Resolve budget parameters
    max_duration = limits.get("maximumDurationSeconds", 60)
    max_searches = limits.get("maximumSearches", 5)
    max_pages = limits.get("maximumPages", 10)
    max_sources = limits.get("maximumSources", 10)
    max_memory = limits.get("maximumMemoryMb") or settings.MAX_MEMORY_MB
    query_domains = None
    has_explicit_domain_allowlist = (
        isinstance(source_policy, dict)
        and isinstance(source_policy.get("allowedDomains"), list)
    )
    if has_explicit_domain_allowlist:
        query_domains = source_policy["allowedDomains"]
    input_chunks, allow_external_inputs = collect_input_context(inputs)
    effective_query, input_limitations = build_effective_query(query, freshness, input_chunks, allow_external_inputs, inputs)
    memory_client = brain_memory_client()
    # Recalled evidence has no per-domain provenance filter in the current Brain
    # contract, so recall is disabled whenever the effective source allowlist is
    # narrower than "anything the profile permits". That covers both an explicit
    # sourcePolicy.allowedDomains and any profile that carries its own allowlist
    # (app/security.py PROFILE_DOMAINS): historical text from a domain the live
    # search would reject must not reach the report.
    profile_rules = PROFILE_DOMAINS.get(profile) or {}
    profile_restricts_domains = bool(profile_rules.get("allowed"))
    # Freshness is the same problem on a different axis. A retained record keeps
    # the claim, origin and source type but no publication date, and
    # recall_context() supplies none either, so "exclude sources published after
    # <until>" cannot be enforced against recalled text — the model would be
    # asked to honour a cutoff it has no dates for. Same remedy as the domain
    # case: withhold recall rather than feed unfiltered history through a filter
    # it cannot satisfy.
    freshness_constrained = any(
        str((freshness or {}).get(key) or "").strip()
        for key in ("since", "until", "maxAgeDays")
    )
    if (
        memory_client
        and settings.BRAIN_MEMORY_CONTEXT_ENABLED
        and not has_explicit_domain_allowlist
        and not profile_restricts_domains
        and not freshness_constrained
    ):
        remaining = max(0.0, max_duration - (time.time() - start_time))
        try:
            historical_context = await asyncio.wait_for(
                asyncio.to_thread(memory_client.recall_context, query),
                timeout=min(3.0, remaining),
            )
        except (asyncio.TimeoutError, ValueError):
            historical_context = ""
        if historical_context:
            effective_query = f"{effective_query}\n\n{historical_context}"

    # Determine gpt-researcher report types based on mode
    report_type = "research_report"
    if mode == "quick":
        report_type = "outline_report"
    elif mode == "deep":
        report_type = "deep"

    max_model_calls = limits.get("maximumModelCalls")
    if max_model_calls is not None and estimate_model_calls(mode) > max_model_calls:
        raise ValueError(
            f"maximumModelCalls={max_model_calls} is too low for {mode} research; "
            f"estimated minimum is {estimate_model_calls(mode)}."
        )

    max_model_cost = limits.get("maximumModelCostUsd")
    if max_model_cost is not None:
        min_cost = estimate_model_cost_usd(estimate_tokens(effective_query), 200)
        if min_cost > max_model_cost:
            raise ValueError(
                f"maximumModelCostUsd=${max_model_cost:.6f} is too low; "
                f"estimated minimum cost for this query is ${min_cost:.6f}."
            )

    env_manager = RequestEnvironmentManager(headers, model_provider=model_provider, model_name=model_name)
    callbacks = GPTResearcherCallbackHandler(reporter)
    execution_duration = max(0.0, max_duration - (time.time() - start_time))
    if execution_duration <= 0:
        raise TimeoutError("maximumDurationSeconds exhausted before research execution")

    with enforce_egress_protection(profile, maximum_searches=max_searches):
        result = await _run_research(
            env_manager,
            callbacks,
            reporter,
            op_id,
            effective_query,
            query,
            mode,
            profile,
            report_type,
            execution_duration,
            max_searches,
            max_pages,
            max_sources,
            max_memory,
            query_domains,
            limits,
            require_claim_verification,
            headers,
            input_chunks,
            input_limitations,
            start_time
        )
    # Outcome ingestion is intentionally NOT scheduled here. The caller
    # (app/api.py background_research_task) persists the result durably first
    # and only then calls schedule_outcome_ingest(), so Brain can never record a
    # completed/partial outcome for a result that was never durably stored.
    return result

async def _run_research(env_manager, callbacks, reporter, op_id, query, display_query, mode, profile, report_type, max_duration, max_searches, max_pages, max_sources, max_memory, query_domains, limits, require_claim_verification, headers, input_chunks, input_limitations, start_time):
    with env_manager.apply_keys():
        await callbacks.on_planning("Initializing research configuration...")

        max_iterations = 2 if mode == "deep" else 1

        researcher = GPTResearcher(
            query=query,
            report_type=report_type,
            query_domains=query_domains
        )

        # maximumSearches is enforced at the outbound search-provider boundary.
        # Keep per-query result/page fanout bounded by source/page limits.
        per_iter_pages = max(1, max_pages // max_iterations)
        researcher.cfg.max_iterations = max_iterations
        researcher.cfg.max_search_results_per_query = max(1, max_sources)
        researcher.cfg.max_urls_per_query = per_iter_pages

        SYNTHESIS_TIMEOUT = min(30, max_duration * 0.25)

        research_task = asyncio.current_task()
        memory_cancelled = False
        lease_lost = False
        client_cancel_event = asyncio.Event()

        async def monitor_memory():
            nonlocal memory_cancelled, lease_lost
            while True:
                await asyncio.sleep(1.0)
                # Heartbeat this instance's ownership lease so a peer instance
                # (or startup reconciliation) can tell the operation is still
                # live and must not be marked stale.
                try:
                    owns = await storage_module.storage.touch_operation(op_id)
                except Exception:
                    # A failing heartbeat means we can no longer prove ownership,
                    # so treat it as lost and fail closed rather than assume the
                    # lease is still ours.
                    logger.warning("Heartbeat for operation %s raised; treating ownership as lost.", op_id, exc_info=True)
                    owns = False
                if owns is False:
                    # Ownership or the concurrency slot was lost (e.g. a long
                    # pause let the lease expire). This is fail-closed: do not
                    # synthesize or persist a result another worker may own.
                    logger.warning("Lost ownership of operation %s during heartbeat; aborting research.", op_id)
                    lease_lost = True
                    research_task.cancel()
                    break
                mem = get_memory_usage_mb()
                if mem > 0.80 * max_memory:
                    logger.warning("Memory threshold exceeded: %.1fMB / %dMB limit. Triggering early synthesis.", mem, max_memory)
                    await reporter.report("synthesizing", f"Memory threshold exceeded ({mem:.1f}MB). Conducting early synthesis.", completed_units=80, total_units=100)
                    memory_cancelled = True
                    research_task.cancel()
                    break

        monitor_task = asyncio.create_task(monitor_memory())

        async def bounded_synthesis(reason: str) -> tuple:
            """Run write_report with a hard time cap. Returns (report_text, status)."""
            try:
                text = await asyncio.wait_for(researcher.write_report(), timeout=SYNTHESIS_TIMEOUT)
                return text, "partial"
            except asyncio.TimeoutError:
                logger.warning("Partial synthesis timed out after %ss for operation %s (%s)", SYNTHESIS_TIMEOUT, op_id, reason)
                return f"Research execution hit {reason}. Partial content could not be fully synthesized within the deadline.", "failed"
            except asyncio.CancelledError:
                if client_cancel_event.is_set():
                    raise
                logger.warning("Partial synthesis cancelled for operation %s (%s)", op_id, reason)
                return f"Research execution hit {reason}. Synthesis was interrupted.", "failed"
            except Exception:
                logger.warning("Partial synthesis failed for operation %s (%s)", op_id, reason, exc_info=True)
                return f"Research execution hit {reason}. Partial content could not be fully synthesized.", "failed"

        try:
            await callbacks.on_planning(f"Starting research loop (budget: {max_duration}s)...")

            async def run_loop():
                await callbacks.on_search(display_query, 1, max(1, max_searches))
                await researcher.conduct_research()
                raw_urls = researcher.get_source_urls() or []
                await callbacks.on_read(
                    raw_urls[0] if raw_urls else "",
                    f"Retrieved {len(raw_urls)} source URL(s).",
                    min(len(raw_urls), max_pages),
                    max(1, max_pages)
                )
                await callbacks.on_synthesize("Synthesizing research report...")
                report = await researcher.write_report()
                # No terminal event here: the caller persists the terminal result
                # and then publishes the single final status, so an SSE consumer
                # can never stop on an early terminal event for a result that was
                # later downgraded (or never stored).
                return report

            report_text = await asyncio.wait_for(run_loop(), timeout=float(max_duration))
            status = "completed"

        except asyncio.TimeoutError:
            logger.warning("Research operation %s hit duration limit of %ss. Synthesizing partial results.", op_id, max_duration)
            await reporter.report("synthesizing", "Research budget exceeded. Synthesizing partial results...")
            report_text, status = await bounded_synthesis("duration limit")

        except asyncio.CancelledError:
            if lease_lost:
                # Fail closed: another worker may own this operation now, so do
                # not synthesize or persist a competing result.
                raise _LeaseLostError(
                    "Operation ownership lease was lost; aborting without writing a result."
                )
            if memory_cancelled:
                logger.warning("Research operation %s cancelled due to memory pressure limit.", op_id)
                await reporter.report("synthesizing", "Memory limit exceeded. Synthesizing partial report.")
                report_text, status = await bounded_synthesis("memory pressure")
            else:
                client_cancel_event.set()
                logger.warning("Research operation %s explicitly cancelled by client.", op_id)
                raise

        except Exception as e:
            if search_budget_exhausted():
                # The run stopped because it used its outbound search allowance.
                # That is a bounded stop, not an execution fault: synthesize the
                # partial report the same way duration/memory limits do.
                logger.warning("Research operation %s exhausted its search budget; synthesizing partial results.", op_id)
                await reporter.report("synthesizing", "Search budget exhausted. Synthesizing partial results...")
                report_text, status = await bounded_synthesis("search budget limit")
            else:
                logger.error("Error during research loop execution: %s", e, exc_info=True)
                raise
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        # Extract structured details from GPT Researcher source/context records.
        raw_sources = researcher.get_source_urls() or []
        search_results = []
        if hasattr(researcher, "get_search_results"):
            try:
                search_results = researcher.get_search_results() or []
            except Exception:
                logger.debug("Unable to read GPT Researcher search results", exc_info=True)

        sources = []
        evidence = []
        claims = []
        citations = []

        # Verify SSRF on each retrieved source before presenting in final result
        safe_sources = []
        for url in raw_sources:
            # Collect up to max_sources *usable* sources: apply the cap after
            # filtering, otherwise provider/unsafe URLs earlier in the list can
            # consume the whole budget and strip valid provenance.
            if len(safe_sources) >= max_sources:
                break
            if is_provider_host(url):
                # A research result must not cite the service's own model/search
                # provider endpoints as web sources.
                logger.warning("Redacting a provider endpoint from final result sources.")
                continue
            if is_safe_url(url, profile):
                safe_sources.append(url)
            else:
                logger.warning("A source URL was flagged by the SSRF filter in the final result. Redacting.")

        source_metadata = collect_source_metadata(researcher, search_results)

        # Build structural schemas
        for idx, url in enumerate(safe_sources):
            source_id = f"src-{op_id}-{idx}"
            metadata = source_metadata.get(url, {})
            sources.append({
                "id": source_id,
                "url": url,
                "title": metadata.get("title") or publisher_from_url(url) or f"Source {idx + 1}",
                "publisher": metadata.get("publisher") or publisher_from_url(url),
                "author": metadata.get("author"),
                "publishedAt": metadata.get("publishedAt"),
                "retrievedAt": int(time.time() * 1000),
                "sourceType": "web",
                "qualityScore": metadata.get("qualityScore")
            })

        output_tokens = estimate_tokens(report_text)
        estimated_input_tokens = estimate_tokens(query) + sum(estimate_tokens(url) for url in safe_sources)
        estimated_cost = estimate_model_cost_usd(estimated_input_tokens, output_tokens)
        budget_reasons = []

        max_model_tokens = limits.get("maximumModelTokens")
        if max_model_tokens is not None and output_tokens > max_model_tokens:
            report_text = trim_to_token_budget(report_text, max_model_tokens)
            output_tokens = estimate_tokens(report_text)
            status = "partial"
            budget_reasons.append(f"maximumModelTokens limited the synthesized answer to approximately {max_model_tokens} tokens.")

        max_model_cost = limits.get("maximumModelCostUsd")
        estimated_cost = estimate_model_cost_usd(estimated_input_tokens, output_tokens)
        if max_model_cost is not None and estimated_cost > max_model_cost:
            status = "partial"
            budget_reasons.append(
                f"estimatedModelCostUsd ${estimated_cost:.6f} exceeded maximumModelCostUsd ${max_model_cost:.6f}."
            )

        passage_records = collect_passage_records(researcher, safe_sources)
        inferred_fallback = False
        if sources and passage_records:
            evidence, claims, citations = build_structured_findings_from_passages(op_id, passage_records, sources)
            claims = verify_claims_against_evidence(op_id, report_text, evidence, citations)
        elif sources:
            # No passage text: emit inferred, unattributed claims rather than
            # fabricating source attribution. Flag the result as degraded.
            evidence, claims, citations = build_structured_findings(op_id, report_text, sources)
            inferred_fallback = True
        elif require_claim_verification:
            claims = [
                {
                    "id": f"claim-{op_id}-0",
                    "text": "No source-backed claims could be verified because GPT Researcher did not expose safe source URLs.",
                    "evidenceIds": [],
                    "confidence": 0.0,
                    "verificationStatus": "unsupported"
                }
            ]

        if input_chunks:
            # Cap total sources before appending local inputs so the combined list stays within the budget.
            del sources[max_sources:]
            append_input_sources(op_id, input_chunks, sources, evidence, citations)
            # Clear claimIds from all citations before re-verifying so stale IDs from the
            # first pass (web-only evidence) don't survive into the final response.
            for citation in citations:
                citation["claimIds"] = []
            claims = verify_claims_against_evidence(op_id, report_text, evidence, citations) or claims
            # Local evidence is now in play. If any final claim is linked to
            # real evidence (supported, partially-supported, or conflicting),
            # the earlier "no passage text / unattributed" web fallback no
            # longer describes the final result, so recompute it after the
            # post-input verification pass instead of relying on the pre-input
            # state or on only the highest support score.
            if inferred_fallback and any(claim.get("evidenceIds") for claim in claims):
                inferred_fallback = False

        degraded_reasons = []
        if search_budget_exhausted():
            degraded_reasons.append("Research stopped at the maximumSearches search-provider budget.")
        if inferred_fallback:
            degraded_reasons.append(
                "No source passage text was available, so claims are report-derived and unattributed rather than source-backed."
            )
        if not sources and not evidence:
            # Neither web sources nor local-input evidence back this report, so
            # it is degraded even though a report was produced. Evaluated after
            # local inputs are appended so input-only evidence is not
            # falsely flagged.
            degraded_reasons.append(
                "No source-backed evidence was available; the report is not supported by any source passage."
            )
        if getattr(storage_module.storage, "degraded", False):
            degraded_reasons.append("Durable storage is degraded; results may not survive a restart.")

        duration_ms = int((time.time() - start_time) * 1000)

        metrics = {
            "startedAt": datetime_from_timestamp(start_time),
            "completedAt": datetime_from_timestamp(time.time()),
            "durationMs": duration_ms,
            "searchesPerformed": len(search_results),
            "pagesRead": len(raw_sources),
            "sourcesConsidered": len(raw_sources),
            "sourcesUsed": len(safe_sources),
            "modelCalls": estimate_model_calls(mode),
            "estimatedModelCostUsd": estimated_cost
        }

        return {
            "operationId": op_id,
            "status": status,
            "mode": mode,
            "profile": profile,
            "answer": report_text,
            "sources": sources,
            "evidence": evidence,
            "claims": claims,
            "citations": citations,
            "searchesPerformed": [res.get("query", "") for res in search_results if isinstance(res, dict)],
            "metrics": metrics,
            "degraded": bool(degraded_reasons),
            "degradedReasons": degraded_reasons or None,
            "limitations": budget_reasons + input_limitations + [
                "Claims are verified by a separate passage-matching pass over extracted evidence.",
                "When GPT Researcher exposes no source passage text, claims are report-derived and left unattributed rather than linked to a source they may not support."
            ]
        }

def datetime_from_timestamp(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat()
