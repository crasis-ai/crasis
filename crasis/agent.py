"""
crasis.agent — CrasisAgent base class for building IO loops around specialists.

A CrasisAgent is the pattern for turning a specialist (or a toolkit of specialists)
into an autonomous process. It:
  1. Polls or receives input from a source
  2. Routes each input through one or more specialists
  3. Takes action based on classification results
  4. Logs decisions locally

Subclass CrasisAgent and implement `fetch_inputs` and `act` to build
any specialist-powered agent: email triagers, support routers,
content moderators, IoT decision loops, etc.

Example:

    class SupportRouter(CrasisAgent):
        def fetch_inputs(self):
            return db.get_unprocessed_tickets()

        def act(self, item, results):
            ticket = item
            urgency = results.get("email-urgency", {}).get("label")
            sentiment = results.get("sentiment-gate", {}).get("label")

            if sentiment == "positive" and urgency == "urgent":
                slack.alert(f"Angry urgent ticket: {ticket['id']}")
                db.route(ticket["id"], queue="escalation")
            elif urgency == "urgent":
                db.route(ticket["id"], queue="priority")
            else:
                db.route(ticket["id"], queue="standard")

    agent = SupportRouter(
        toolkit=CrasisToolkit.from_dir("./models"),
        specialists=["email-urgency", "sentiment-gate"],
    )
    agent.run()
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from crasis.tools import CrasisToolkit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision record
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    """
    A single agent decision — one input processed through all specialists.

    Attributes:
        input_id: Optional identifier for the input item.
        text: The text that was classified.
        results: Dict of specialist_name → classification result.
        action: The action key returned by act().
        latency_ms: Total processing time for this decision.
    """

    input_id: Any
    text: str
    results: dict[str, dict]
    action: str | None
    latency_ms: float


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------


class CrasisAgent(ABC):
    """
    Base class for Crasis specialist agents.

    Subclass this and implement `fetch_inputs` and `act`.
    Everything else — routing, error handling, loop timing — is handled.

    Args:
        toolkit: Loaded CrasisToolkit with all required specialists.
        specialists: List of specialist names to run on every input.
                     If None, runs all specialists in the toolkit.
        poll_interval_s: Seconds to sleep between polling cycles.
                         Set to 0 for tight loops (e.g. stream processing).
        on_decision: Optional callback called with each Decision after act().
        max_errors: Stop the agent after this many consecutive errors.
    """

    def __init__(
        self,
        toolkit: CrasisToolkit,
        specialists: list[str] | None = None,
        poll_interval_s: float = 1.0,
        on_decision: Callable[[Decision], None] | None = None,
        max_errors: int = 10,
    ) -> None:
        self.toolkit = toolkit
        self._specialist_names = specialists or toolkit.specialists()
        self._poll_interval = poll_interval_s
        self._on_decision = on_decision
        self._max_errors = max_errors
        self._running = False
        self._decisions: list[Decision] = []
        self._consecutive_errors = 0

        # Validate that all requested specialists are available
        available = set(toolkit.specialists())
        missing = set(self._specialist_names) - available
        if missing:
            raise ValueError(
                f"Specialists not found in toolkit: {missing}. "
                f"Available: {available}"
            )

        logger.info(
            "%s initialized with specialists: %s",
            self.__class__.__name__,
            self._specialist_names,
        )

    # ------------------------------------------------------------------
    # Abstract interface — subclass must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch_inputs(self) -> list[dict]:
        """
        Return a list of input items to process.

        Each item should be a dict with at minimum:
          - "text": str — the text to classify
          - "id": any — optional identifier for logging and act()

        Called on every polling cycle. Return an empty list if there's
        nothing to process — the agent will sleep and try again.

        Returns:
            List of input dicts. Empty list = nothing to do this cycle.
        """
        ...

    @abstractmethod
    def act(self, item: dict, results: dict[str, dict]) -> str | None:
        """
        Take action based on classification results.

        Called once per input item, after all specialists have run.

        Args:
            item: The original input dict from fetch_inputs.
            results: Dict of specialist_name → classification result dict.
                     e.g. {"sentiment-gate": {"label": "positive", "confidence": 0.97, ...}}

        Returns:
            A string describing the action taken (for logging), or None.
        """
        ...

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self, max_cycles: int | None = None) -> None:
        """
        Start the agent's main loop.

        Polls fetch_inputs, runs specialists, calls act, logs decisions.
        Runs until stop() is called, max_cycles is reached, or max_errors
        consecutive failures occur.

        Args:
            max_cycles: Stop after this many polling cycles. None = run forever.
        """
        self._running = True
        cycle = 0
        total_processed = 0

        logger.info("%s starting run loop", self.__class__.__name__)

        try:
            while self._running:
                if max_cycles is not None and cycle >= max_cycles:
                    logger.info("Reached max_cycles=%d, stopping", max_cycles)
                    break

                try:
                    inputs = self.fetch_inputs()
                    self._consecutive_errors = 0
                except Exception as exc:
                    self._consecutive_errors += 1
                    logger.error(
                        "fetch_inputs failed (error %d/%d): %s",
                        self._consecutive_errors, self._max_errors, exc,
                    )
                    if self._consecutive_errors >= self._max_errors:
                        raise RuntimeError(
                            f"Agent stopped: {self._max_errors} consecutive fetch_inputs failures"
                        ) from exc
                    time.sleep(self._poll_interval)
                    cycle += 1
                    continue

                if not inputs:
                    time.sleep(self._poll_interval)
                    cycle += 1
                    continue

                for item in inputs:
                    if not self._running:
                        break
                    decision = self._process_item(item)
                    self._decisions.append(decision)
                    total_processed += 1

                    if self._on_decision:
                        try:
                            self._on_decision(decision)
                        except Exception as exc:
                            logger.warning("on_decision callback failed: %s", exc)

                cycle += 1
                if self._poll_interval > 0:
                    time.sleep(self._poll_interval)

        except KeyboardInterrupt:
            logger.info("Agent interrupted by user")
        finally:
            self._running = False
            logger.info(
                "%s stopped. Processed %d items across %d cycles.",
                self.__class__.__name__, total_processed, cycle,
            )

    def stop(self) -> None:
        """Signal the agent to stop after the current cycle."""
        self._running = False

    # ------------------------------------------------------------------
    # Stats and introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return runtime statistics."""
        if not self._decisions:
            return {"total_decisions": 0}

        latencies = [d.latency_ms for d in self._decisions]
        actions = {}
        for d in self._decisions:
            actions[d.action or "none"] = actions.get(d.action or "none", 0) + 1

        return {
            "total_decisions": len(self._decisions),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "actions": actions,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_item(self, item: dict) -> Decision:
        """Run all specialists on one item and call act()."""
        t0 = time.perf_counter()
        text = item.get("text", "")
        input_id = item.get("id")

        # Run all specialists
        results: dict[str, dict] = {}
        for name in self._specialist_names:
            try:
                results[name] = self.toolkit.classify(name, text)
            except Exception as exc:
                logger.warning("Specialist '%s' failed on item %s: %s", name, input_id, exc)
                results[name] = {"error": str(exc)}

        # Call act
        try:
            action = self.act(item, results)
        except Exception as exc:
            logger.error("act() failed for item %s: %s", input_id, exc)
            action = "error"

        latency_ms = (time.perf_counter() - t0) * 1000

        logger.debug(
            "Item %s → action=%s (%.1fms)",
            input_id, action, latency_ms,
        )

        return Decision(
            input_id=input_id,
            text=text,
            results=results,
            action=action,
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Backwards-compatible alias
# ---------------------------------------------------------------------------

ForgeAgent = CrasisAgent
