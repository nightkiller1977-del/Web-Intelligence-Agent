# app/metrics.py
import logging
from prometheus_client import Counter, Histogram
from app.config import settings

logger = logging.getLogger("web-intelligence")

# Prometheus Metrics
research_cost_tokens = Counter(
    "research_cost_tokens_total",
    "Total tokens consumed during web intelligence operations",
    ["agent_profile", "token_type"] # e.g. prompt_tokens, completion_tokens
)

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

# Ephemeral Daily Spend Accumulator (In-Memory for simplicity, or Redis if configured)
_daily_spend_usd = 0.0

def track_operation_cost(agent_profile: str, input_tokens: int, output_tokens: int, provider: str = "openai"):
    global _daily_spend_usd
    
    # Very rough cost calculations based on average LLM pricing (e.g. $2.50 / 1M input, $10.00 / 1M output)
    input_rate = 0.0000025
    output_rate = 0.000010
    
    cost = (input_tokens * input_rate) + (output_tokens * output_rate)
    
    # Increment Prometheus metrics
    research_cost_tokens.labels(agent_profile=agent_profile, token_type="input").inc(input_tokens)
    research_cost_tokens.labels(agent_profile=agent_profile, token_type="output").inc(output_tokens)
    
    _daily_spend_usd += cost
    logger.info(f"Operation cost: ${cost:.5f} USD. Accumulated Daily Spend: ${_daily_spend_usd:.2f} USD")
    
    if _daily_spend_usd > settings.DAILY_SPEND_LIMIT_USD:
        logger.warning(
            f"DAILY SPEND LIMIT EXCEEDED! Current: ${_daily_spend_usd:.2f} USD, Limit: {settings.DAILY_SPEND_LIMIT_USD:.2f} USD. "
            "Please check for runaway loops or adjust limits."
        )

def get_accumulated_daily_spend():
    return _daily_spend_usd
