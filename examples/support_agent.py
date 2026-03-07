"""
examples/support_agent.py — Autonomous support ticket router using CrasisAgent.

Demonstrates the full IO loop pattern:
  1. Polls for unprocessed tickets
  2. Runs each through sentiment-gate + email-urgency
  3. Routes to the correct queue based on combined classification
  4. Logs all decisions locally

This is a self-contained demo — it generates synthetic tickets and processes
them in memory. Swap fetch_inputs() for your real ticket source.

Usage:
    python examples/support_agent.py
"""

from __future__ import annotations

import time
from crasis.agent import CrasisAgent
from crasis.tools import CrasisToolkit


# ---------------------------------------------------------------------------
# Synthetic ticket source (replace with your real source)
# ---------------------------------------------------------------------------

SYNTHETIC_TICKETS = [
    {"id": "T001", "text": "I WANT MY MONEY BACK RIGHT NOW. This is completely unacceptable!!!"},
    {"id": "T002", "text": "Hi, just checking on the status of my order #4521. Thanks!"},
    {"id": "T003", "text": "Your product broke after ONE DAY. I'm disputing this with my bank."},
    {"id": "T004", "text": "Could you clarify the return policy? I might need to send something back."},
    {"id": "T005", "text": "WORST CUSTOMER SERVICE EVER. I've been waiting 3 weeks for a response."},
    {"id": "T006", "text": "The packaging was a bit damaged but the product seems fine. Just wanted to flag it."},
    {"id": "T007", "text": "How do I update my billing address?"},
    {"id": "T008", "text": "I am SO DONE with this company. Cancelling everything. Goodbye."},
    {"id": "T009", "text": "Love the product! Just wondering if there's a loyalty program?"},
    {"id": "T010", "text": "Still haven't received my refund from 2 weeks ago. What is going on?!"},
]


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------

class SupportRouter(CrasisAgent):
    """
    Routes support tickets based on urgency and sentiment.

    Routing logic:
      angry + urgent  -> escalation queue (immediate human review)
      angry + routine -> priority queue
      calm  + urgent  -> standard queue (but flagged)
      calm  + routine -> standard queue
    """

    def __init__(self, toolkit: CrasisToolkit) -> None:
        super().__init__(
            toolkit=toolkit,
            specialists=["sentiment-gate"],
            poll_interval_s=0,  # process as fast as possible
        )
        self._ticket_queue = list(SYNTHETIC_TICKETS)
        self._routing: dict[str, list] = {
            "escalation": [],
            "priority": [],
            "standard": [],
        }

    def fetch_inputs(self) -> list[dict]:
        """Return one ticket per cycle, or empty list when done."""
        if not self._ticket_queue:
            self.stop()
            return []
        # Simulate real polling — return one at a time
        return [self._ticket_queue.pop(0)]

    def act(self, item: dict, results: dict) -> str:
        """Route the ticket based on sentiment classification."""
        sentiment = results.get("sentiment-gate", {})
        is_angry = sentiment.get("label") == "positive"
        confidence = sentiment.get("confidence", 0.0)

        # Simple routing — in production, combine multiple specialists here
        if is_angry and confidence >= 0.85:
            queue = "escalation"
        elif is_angry:
            queue = "priority"
        else:
            queue = "standard"

        self._routing[queue].append(item["id"])
        return queue


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("Crasis Support Router Demo")
    print("-" * 50)
    print("Loading specialists from ./models/...")
    print()

    try:
        toolkit = CrasisToolkit.from_dir("./models")
    except FileNotFoundError:
        print("No models found in ./models/")
        print("   Run `crasis build --spec specialists/sentiment-gate/spec.yaml` first.")
        print()
        print("Running in DEMO MODE with mock classifications...\n")
        _run_mock_demo()
        return

    agent = SupportRouter(toolkit)

    # Wire up a live decision logger
    def log_decision(decision):
        sentiment = decision.results.get("sentiment-gate", {})
        label = sentiment.get("label", "?")
        conf = sentiment.get("confidence", 0)
        print(
            f"  [{decision.input_id}] → {decision.action:12s} "
            f"| sentiment={label} ({conf:.2f}) "
            f"| {decision.latency_ms:.1f}ms"
        )

    agent._on_decision = log_decision

    t0 = time.perf_counter()
    agent.run()
    total_ms = (time.perf_counter() - t0) * 1000

    stats = agent.stats()
    routing = agent._routing

    print()
    print("─" * 50)
    print(f"Processed {stats['total_decisions']} tickets in {total_ms:.0f}ms")
    print(f"Avg per ticket: {stats['avg_latency_ms']:.1f}ms")
    print()
    print("Routing results:")
    print(f"  Escalation : {routing['escalation']}")
    print(f"  Priority   : {routing['priority']}")
    print(f"  Standard   : {routing['standard']}")
    print()
    print("Zero API calls made. Model ran locally.")


def _run_mock_demo() -> None:
    """Demo output when no trained model is available yet."""

    mock_results = [
        ("T001", "escalation", "positive", 0.97, 42),
        ("T002", "standard",   "negative", 0.95, 38),
        ("T003", "escalation", "positive", 0.99, 41),
        ("T004", "standard",   "negative", 0.92, 37),
        ("T005", "escalation", "positive", 0.98, 40),
        ("T006", "standard",   "negative", 0.89, 39),
        ("T007", "standard",   "negative", 0.96, 38),
        ("T008", "priority",   "positive", 0.81, 43),
        ("T009", "standard",   "negative", 0.97, 37),
        ("T010", "priority",   "positive", 0.83, 41),
    ]

    routing: dict[str, list] = {"escalation": [], "priority": [], "standard": []}
    total_ms = 0

    for ticket_id, queue, label, conf, latency in mock_results:
        print(
            f"  [{ticket_id}] → {queue:12s} "
            f"| sentiment={label} ({conf:.2f}) "
            f"| {latency}ms"
        )
        routing[queue].append(ticket_id)
        total_ms += latency
        time.sleep(0.05)  # make it feel real

    print()
    print("─" * 50)
    print(f"Processed 10 tickets in {total_ms}ms total")
    print(f"Avg per ticket: {total_ms / 10:.1f}ms")
    print()
    print("Routing results:")
    print(f"  Escalation : {routing['escalation']}")
    print(f"  Priority   : {routing['priority']}")
    print(f"  Standard   : {routing['standard']}")
    print()
    print("(Mock demo — train the model to see real results)")


if __name__ == "__main__":
    main()
