from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# 1. Retrieval latency — how long the vector search takes (seconds)
retrieval_latency = Histogram(
    "rag_retrieval_latency_seconds",
    "Time spent on vector search"
)

# 2. Token usage — total tokens used (prompt + completion), labeled by type
tokens_used = Counter(
    "rag_tokens_used_total",
    "Total tokens used by the LLM",
    ["kind"]  # "prompt" or "completion"
)

# 3. Cost — dollars spent, accumulated over time
cost_dollars = Counter(
    "rag_cost_dollars_total",
    "Total LLM cost in USD"
)

# 4. Query counter — how many /ask calls, labeled by outcome
queries_total = Counter(
    "rag_queries_total",
    "Total number of questions asked",
    ["outcome"]  # "answered" or "refused"
)

def metrics_endpoint():
    return generate_latest(), CONTENT_TYPE_LATEST