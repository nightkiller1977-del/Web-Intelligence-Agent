# app/metrics.py
import logging

from prometheus_client import Counter, Histogram

logger = logging.getLogger("web-intelligence")

research_duration = Histogram(
    "research_duration_ms",
    "Research execution duration in milliseconds",
    ["agent_profile", "mode"],
    buckets=(1000, 5000, 10000, 30000, 60000, 120000, 180000)
)

sources_fetched = Histogram(
    "sources_fetched_count",
    "Count of web sources fetched per research operation",
    ["agent_profile"],
    buckets=(1, 3, 5, 10, 20, 50)
)

research_operations = Counter(
    "research_operations_total",
    "Research operations by final status",
    ["agent_profile", "mode", "status"]
)

research_cost_tokens = Counter(
    "research_cost_tokens_total",
    "Estimated output tokens consumed during web intelligence operations",
    ["agent_profile", "token_type"]
)

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()) * 4 // 3)

def track_operation_cost(agent_profile: str, output_tokens: int):
    # Cost enforcement lives entirely in the shared, atomic reservation in
    # storage. A process-local counter here would reset on restart and grant
    # every replica a separate allowance, so only the metric is recorded.
    research_cost_tokens.labels(agent_profile=agent_profile, token_type="output").inc(output_tokens)

def observed_result_cost(result: dict) -> float:
    """Estimated spend for a completed result, falling back to a text estimate."""
    metrics = result.get("metrics") or {}
    cost = metrics.get("estimatedModelCostUsd")
    if cost is None:
        cost = estimate_tokens(result.get("answer") or "") * 0.000010
    return float(cost or 0.0)


async def record_operation_spend(storage, result: dict) -> None:
    """Accumulate this result's estimated spend in shared storage.

    Admission enforcement lives entirely in the shared atomic reservation; this
    records the observed cost so the shared window stays a faithful running
    total across replicas.
    """
    metrics = result.get("metrics") or {}
    cost = metrics.get("estimatedModelCostUsd")
    if cost is None:
        cost = estimate_tokens(result.get("answer") or "") * 0.000010
    if cost:
        try:
            await storage.add_daily_spend(float(cost))
        except Exception:
            logger.warning("Failed to record estimated spend in shared storage; using process-local estimate only.")


def observe_research_result(result: dict):
    profile = result.get("profile", "unknown")
    mode = result.get("mode", "unknown")
    status = result.get("status", "unknown")
    metrics = result.get("metrics") or {}
    sources = result.get("sources") or []

    research_operations.labels(agent_profile=profile, mode=mode, status=status).inc()
    research_duration.labels(agent_profile=profile, mode=mode).observe(metrics.get("durationMs", 0))
    sources_fetched.labels(agent_profile=profile).observe(len(sources))

    output_tokens = estimate_tokens(result.get("answer") or "")
    if output_tokens:
        track_operation_cost(profile, output_tokens)
