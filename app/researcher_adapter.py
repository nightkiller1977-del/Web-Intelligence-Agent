# app/researcher_adapter.py
import asyncio
import logging
import time
from typing import Dict, Any
from gpt_researcher import GPTResearcher

from app.progress_adapter import ProgressReporter, GPTResearcherCallbackHandler
from app.model_adapter import RequestEnvironmentManager
from app.security import enforce_egress_protection, is_safe_url
from app.config import settings

logger = logging.getLogger("web-intelligence")
UNSUPPORTED_MODEL_LIMIT_FIELDS = (
    "maximumModelCalls",
    "maximumModelTokens",
    "maximumModelCostUsd",
)

import psutil
import os

def get_memory_usage_mb() -> float:
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0

async def conduct_web_research(
    op_id: str,
    query: str,
    mode: str,
    profile: str,
    limits: Dict[str, Any],
    reporter: ProgressReporter,
    headers: Dict[str, str]
) -> Dict[str, Any]:
    start_time = time.time()

    unsupported_limits = [
        field for field in UNSUPPORTED_MODEL_LIMIT_FIELDS
        if limits.get(field) is not None
    ]
    if unsupported_limits:
        raise ValueError(
            "Unsupported model budget limits: "
            f"{', '.join(unsupported_limits)}"
        )

    # Resolve budget parameters
    max_duration = limits.get("maximumDurationSeconds", 60)
    max_searches = limits.get("maximumSearches", 5)
    max_pages = limits.get("maximumPages", 10)
    max_sources = limits.get("maximumSources", 10)
    max_memory = limits.get("maximumMemoryMb") or settings.MAX_MEMORY_MB

    # Determine gpt-researcher report types based on mode
    report_type = "research_report"
    if mode == "quick":
        report_type = "outline_report"
    elif mode == "deep":
        report_type = "deep"

    env_manager = RequestEnvironmentManager(headers)
    callbacks = GPTResearcherCallbackHandler(reporter)

    with enforce_egress_protection(profile):
        return await _run_research(env_manager, callbacks, reporter, op_id, query, mode, profile, report_type, max_duration, max_searches, max_pages, max_sources, max_memory, headers)

async def _run_research(env_manager, callbacks, reporter, op_id, query, mode, profile, report_type, max_duration, max_searches, max_pages, max_sources, max_memory, headers):
    with env_manager.apply_keys():
        await callbacks.on_planning("Initializing research configuration...")

        max_iterations = 2 if mode == "deep" else 1

        researcher = GPTResearcher(
            query=query,
            report_type=report_type
        )

        # Distribute the caller's total budget across iterations so the
        # aggregate stays within the requested maximumSearches / maximumPages.
        per_iter_searches = max(1, max_searches // max_iterations)
        per_iter_pages = max(1, max_pages // max_iterations)
        researcher.cfg.max_iterations = max_iterations
        researcher.cfg.max_search_results_per_query = per_iter_searches
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

        # Extract structured details. GPT Researcher exposes URLs reliably across
        # versions, but evidence passages are not guaranteed through a stable API.
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

        # Build structural schemas
        for idx, url in enumerate(safe_sources):
            source_id = f"src-{op_id}-{idx}"
            sources.append({
                "id": source_id,
                "url": url,
                "title": f"Source {idx + 1}",
                "retrievedAt": int(time.time() * 1000),
                "sourceType": "web"
            })

        duration_ms = int((time.time() - start_time) * 1000)

        metrics = {
            "startedAt": datetime_from_timestamp(start_time),
            "completedAt": datetime_from_timestamp(time.time()),
            "durationMs": duration_ms,
            "searchesPerformed": len(search_results),
            "pagesRead": len(raw_sources),
            "sourcesConsidered": len(raw_sources),
            "sourcesUsed": len(safe_sources)
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
            "limitations": [
                "Structured evidence and claim verification were not emitted because the current GPT Researcher adapter exposes source URLs but not stable passage-level evidence."
            ]
        }

def datetime_from_timestamp(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat()
