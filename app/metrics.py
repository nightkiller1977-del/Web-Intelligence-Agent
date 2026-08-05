# app/metrics.py
from prometheus_client import Histogram

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
