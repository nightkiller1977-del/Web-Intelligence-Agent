# app/researcher_adapter.py
import asyncio
import logging
import time
from typing import Dict, Any, List
from gpt_researcher import GPTResearcher

from app.progress_adapter import ProgressReporter, GPTResearcherCallbackHandler
from app.model_adapter import RequestEnvironmentManager
from app.security import is_safe_url
from app.config import settings

logger = logging.getLogger("web-intelligence")

import sys
import resource

def get_memory_usage_mb() -> float:
    try:
        # ru_maxrss returns bytes on macOS, kilobytes on Linux
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            return rss / (1024 * 1024)
        else:
            return rss / 1024
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
        report_type = "detailed_report"

    env_manager = RequestEnvironmentManager(headers)
    callbacks = GPTResearcherCallbackHandler(reporter)

    # We run gpt-researcher inside the request env context
    with env_manager.apply_keys():
        await callbacks.on_planning("Initializing research configuration...")

        # Initialize GPT Researcher
        researcher = GPTResearcher(
            query=query,
            report_type=report_type,
            max_iterations=2 if mode == "deep" else 1
        )

        # Enforce budget overrides (e.g. limiting search iterations and pages)
        researcher.cfg.max_search_results_per_query = max_searches
        researcher.cfg.max_urls_per_query = max_pages

        # Setup memory monitoring task
        research_task = asyncio.current_task()
        memory_cancelled = False

        async def monitor_memory():
            nonlocal memory_cancelled
            while True:
                await asyncio.sleep(1.0)
                mem = get_memory_usage_mb()
                if mem > 0.80 * max_memory:
                    logger.warning(f"Memory threshold exceeded: {mem:.1f}MB / {max_memory}MB limit. Triggering early synthesis.")
                    await reporter.report("synthesizing", f"Memory threshold exceeded ({mem:.1f}MB). Conducting early synthesis.", completed_units=80, total_units=100)
                    memory_cancelled = True
                    research_task.cancel()
                    break

        monitor_task = asyncio.create_task(monitor_memory())

        # Run execution with timeout and memory checks
        try:
            await callbacks.on_planning(f"Starting research loop (budget: {max_duration}s)...")

            async def run_loop():
                await researcher.conduct_research()
                report = await researcher.write_report()
                return report

            # Run researcher with dynamic budget timeout
            report_text = await asyncio.wait_for(run_loop(), timeout=float(max_duration))
            status = "completed"

        except asyncio.TimeoutError:
            logger.warning(f"Research operation {op_id} hit duration limit of {max_duration}s. Synthesizing partial results.")
            await reporter.report("synthesizing", "Research budget exceeded. Synthesizing partial results...")
            try:
                report_text = await asyncio.shield(researcher.write_report())
                status = "partial"
            except Exception as write_err:
                logger.warning("Partial synthesis failed after timeout for operation %s", op_id, exc_info=True)
                report_text = "Research execution timed out. Partial content could not be fully synthesized."
                status = "failed"

        except asyncio.CancelledError as e:
            if memory_cancelled:
                logger.warning(f"Research operation {op_id} cancelled due to memory pressure limit.")
                await reporter.report("synthesizing", "Memory limit exceeded. Spilling current context buffers to disk and synthesizing partial report.")
                try:
                    report_text = await asyncio.shield(researcher.write_report())
                    status = "partial"
                except Exception as write_err:
                    logger.warning("Partial synthesis failed after memory pressure for operation %s", op_id, exc_info=True)
                    report_text = "Research execution hit the memory limit. Partial content could not be fully synthesized."
                    status = "failed"
            else:
                logger.warning(f"Research operation {op_id} explicitly cancelled by client.")
                raise e

        except Exception as e:
            logger.error(f"Error during research loop execution: {e}", exc_info=True)
            raise e
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        # Extract structured details. GPT Researcher exposes URLs reliably across
        # versions, but evidence passages are not guaranteed through a stable API.
        raw_sources = researcher.get_source_urls()
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
