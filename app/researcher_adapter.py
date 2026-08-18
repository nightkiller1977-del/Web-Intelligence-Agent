# app/researcher_adapter.py
import asyncio
import re
import logging
import time
import hashlib
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, Any
from gpt_researcher import GPTResearcher

from app.progress_adapter import ProgressReporter, GPTResearcherCallbackHandler
from app.model_adapter import RequestEnvironmentManager
from app.security import enforce_egress_protection, is_safe_url
from app.config import settings

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
    return [candidate.strip() for candidate in candidates if len(candidate.strip()) >= 24]

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
    sentences = split_sentences(text)
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
        source = source_by_url.get(record.get("url")) or default_source
        if not source:
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

    for claim_text in split_sentences(report_text)[:maximum_claims]:
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
    evidence = []
    claims = []
    citations_by_source = {
        source["id"]: {
            "id": f"cite-{op_id}-{idx}",
            "sourceId": source["id"],
            "evidenceIds": [],
            "claimIds": []
        }
        for idx, source in enumerate(sources)
    }

    sentences = split_sentences(report_text)[:maximum_items]
    for idx, sentence in enumerate(sentences):
        source = sources[idx % len(sources)] if sources else None
        evidence_id = f"ev-{op_id}-{idx}"
        claim_id = f"claim-{op_id}-{idx}"

        if source:
            evidence.append({
                "id": evidence_id,
                "sourceId": source["id"],
                "passage": sentence,
                "contentHash": content_hash(sentence),
                "relevanceScore": 0.8
            })
            evidence_ids = [evidence_id]
            verification_status = "partially-supported"
            citations_by_source[source["id"]]["evidenceIds"].append(evidence_id)
            citations_by_source[source["id"]]["claimIds"].append(claim_id)
        else:
            evidence_ids = []
            verification_status = "inferred"

        claims.append({
            "id": claim_id,
            "text": sentence,
            "evidenceIds": evidence_ids,
            "confidence": 0.7 if evidence_ids else 0.45,
            "verificationStatus": verification_status
        })

    return evidence, claims, list(citations_by_source.values())

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

def input_text_from_file(path: Path) -> str:
    if settings.DEPLOYMENT_MODE != "local":
        return ""
    path = path.resolve()
    if not path.is_file() or path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
        return ""
    try:
        with path.open("rb") as input_file:
            raw = input_file.read(MAX_INPUT_FILE_BYTES)
        return raw.decode("utf-8", errors="replace")
    except OSError:
        logger.warning("Unable to read input file: %s", path)
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
        root = Path(str(item["path"])).expanduser()
        if not root.is_dir():
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

def format_input_context_for_query(chunks: list[dict]) -> str:
    sections = []
    for chunk in chunks:
        sections.append(f"Input: {chunk['label']}\n{chunk['text']}")
    return "\n\n---\n\n".join(sections)

def build_effective_query(query: str, freshness: Dict[str, str] | None, input_chunks: list[dict], allow_external_inputs: bool) -> tuple[str, list[str]]:
    additions = []
    limitations = []
    fresh = freshness_instruction(freshness)
    if fresh:
        additions.append(f"Freshness constraint: {fresh}.")
    if input_chunks:
        limitations.append("Local document/repository inputs were processed as bounded first-party evidence.")
        if allow_external_inputs:
            additions.append("Use this explicitly provided local input context as first-party context:\n" + format_input_context_for_query(input_chunks))
            limitations.append("Local inputs were explicitly allowed for external research prompt context.")
        else:
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
            "url": f"file://{chunk['path']}",
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
    if isinstance(source_policy, dict) and isinstance(source_policy.get("allowedDomains"), list):
        query_domains = source_policy["allowedDomains"]
    input_chunks, allow_external_inputs = collect_input_context(inputs)
    effective_query, input_limitations = build_effective_query(query, freshness, input_chunks, allow_external_inputs)

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

    with enforce_egress_protection(profile, maximum_searches=max_searches):
        return await _run_research(
            env_manager,
            callbacks,
            reporter,
            op_id,
            effective_query,
            mode,
            profile,
            report_type,
            max_duration,
            max_searches,
            max_pages,
            max_sources,
            max_memory,
            query_domains,
            limits,
            require_claim_verification,
            headers,
            input_chunks,
            input_limitations
        )

async def _run_research(env_manager, callbacks, reporter, op_id, query, mode, profile, report_type, max_duration, max_searches, max_pages, max_sources, max_memory, query_domains, limits, require_claim_verification, headers, input_chunks, input_limitations):
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
        client_cancel_event = asyncio.Event()

        async def monitor_memory():
            nonlocal memory_cancelled
            while True:
                await asyncio.sleep(1.0)
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
                await callbacks.on_search(query, 1, max(1, max_searches))
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
                await reporter.report("completed", "Research completed.", completed_units=100, total_units=100)
                return report

            report_text = await asyncio.wait_for(run_loop(), timeout=float(max_duration))
            status = "completed"

        except asyncio.TimeoutError:
            logger.warning("Research operation %s hit duration limit of %ss. Synthesizing partial results.", op_id, max_duration)
            await reporter.report("synthesizing", "Research budget exceeded. Synthesizing partial results...")
            report_text, status = await bounded_synthesis("duration limit")

        except asyncio.CancelledError:
            if memory_cancelled:
                logger.warning("Research operation %s cancelled due to memory pressure limit.", op_id)
                await reporter.report("synthesizing", "Memory limit exceeded. Synthesizing partial report.")
                report_text, status = await bounded_synthesis("memory pressure")
            else:
                client_cancel_event.set()
                logger.warning("Research operation %s explicitly cancelled by client.", op_id)
                raise

        except Exception as e:
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
        for i, url in enumerate(raw_sources):
            if i >= max_sources:
                break
            if is_safe_url(url, profile):
                safe_sources.append(url)
            else:
                logger.warning(f"Source URL {url} flagged by SSRF filter in final result. Redacting.")

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
        if sources and passage_records:
            evidence, claims, citations = build_structured_findings_from_passages(op_id, passage_records, sources)
            claims = verify_claims_against_evidence(op_id, report_text, evidence, citations)
        elif sources:
            evidence, claims, citations = build_structured_findings(op_id, report_text, sources)
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
            "limitations": budget_reasons + input_limitations + [
                "Claims are verified by a separate passage-matching pass over extracted evidence.",
                "If GPT Researcher exposes no source text for a safe URL, the adapter falls back to report-derived inferred claims for that source."
            ]
        }

def datetime_from_timestamp(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat()
