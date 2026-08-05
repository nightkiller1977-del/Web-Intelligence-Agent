# app/metrics.py
import logging

from prometheus_client import Counter, Histogram

from app.config import settings

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

_daily_spend_usd = 0.0

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()) * 4 // 3)

def track_operation_cost(agent_profile: str, output_tokens: int):
    global _daily_spend_usd

    # Rough default estimate: $10 per 1M output tokens.
    output_rate = 0.000010
    cost = output_tokens * output_rate

    research_cost_tokens.labels(agent_profile=agent_profile, token_type="output").inc(output_tokens)
    _daily_spend_usd += cost

    if _daily_spend_usd > settings.DAILY_SPEND_LIMIT_USD:
        logger.warning(
            "Daily web intelligence spend estimate exceeded limit: current=$%.2f limit=$%.2f",
            _daily_spend_usd,
            settings.DAILY_SPEND_LIMIT_USD
        )

def get_accumulated_daily_spend():
    return _daily_spend_usd

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
