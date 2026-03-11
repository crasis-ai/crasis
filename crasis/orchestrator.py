"""
crasis.orchestrator — Agentic loop orchestrator for OpenAI, Anthropic, and Gemini.

Wraps the multi-turn tool-calling loop so users don't have to wire it manually.
Specialists run locally. The frontier model handles reasoning only.

Usage:
    from crasis.tools import CrasisToolkit
    from crasis.orchestrator import CrasisOrchestrator

    toolkit = CrasisToolkit.from_dir("~/.crasis/specialists")
    orch = CrasisOrchestrator(toolkit=toolkit, provider="anthropic", model="claude-opus-4-5")
    result = orch.run("Is this customer angry? 'I WANT A REFUND NOW'")
    print(result.response)
    print(result.tool_calls)  # [(tool_name, input_text, result_dict), ...]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from crasis.tools import CrasisToolkit


@dataclass
class OrchestratorResult:
    """
    Result of a single orchestrator run.

    Attributes:
        response: Final text response from the frontier model.
        tool_calls: List of (tool_name, input_text, result_dict) tuples for each
                    specialist that was invoked during the run.
        total_latency_ms: Wall-clock time for the entire run in milliseconds.
        frontier_model: Model ID used for the run.
        provider: Provider name ("openai", "anthropic", or "gemini").
    """

    response: str
    tool_calls: list[tuple[str, str, dict]] = field(default_factory=list)
    total_latency_ms: float = 0.0
    frontier_model: str = ""
    provider: str = ""


class CrasisOrchestrator:
    """
    Convenience orchestrator that wires the agentic tool-calling loop for
    OpenAI, Anthropic, and Gemini.

    Each provider loop:
    1. Sends the user prompt with all loaded specialist tools advertised.
    2. Executes any tool calls locally using the toolkit.
    3. Feeds results back to the model.
    4. Repeats until the model returns a final text response or max_iterations is hit.

    The frontier model only sees text classification results — the raw model
    weights never leave the local machine.
    """

    def __init__(
        self,
        toolkit: CrasisToolkit,
        provider: Literal["openai", "anthropic", "gemini"],
        model: str,
        api_key: str | None = None,
        max_iterations: int = 10,
        system_prompt: str | None = None,
    ) -> None:
        """
        Args:
            toolkit: Loaded CrasisToolkit instance.
            provider: Which frontier model provider to use.
            model: Model ID string (e.g. "claude-opus-4-5", "gpt-4o", "gemini-pro").
            api_key: API key for the provider. Falls back to provider-standard env vars.
            max_iterations: Maximum number of tool-call rounds before aborting.
            system_prompt: Optional system prompt. Defaults to a brief Crasis description.
        """
        self._toolkit = toolkit
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._max_iterations = max_iterations
        self._system_prompt = system_prompt or (
            "You are a helpful assistant with access to local Crasis specialist models. "
            "Use the classification tools to answer questions about text. "
            "Specialists run locally in <100ms with no cloud API calls."
        )

    def run(self, prompt: str) -> OrchestratorResult:
        """
        Run the agentic loop for a single user prompt.

        Args:
            prompt: User message to process.

        Returns:
            OrchestratorResult with the final response and all tool calls made.

        Raises:
            ImportError: If the required provider SDK is not installed.
        """
        t0 = time.perf_counter()

        if self._provider == "anthropic":
            result = self._run_anthropic_loop(prompt)
        elif self._provider == "openai":
            result = self._run_openai_loop(prompt)
        elif self._provider == "gemini":
            result = self._run_gemini_loop(prompt)
        else:
            raise ValueError(
                f"Unknown provider '{self._provider}'. "
                "Must be one of: 'openai', 'anthropic', 'gemini'."
            )

        result.total_latency_ms = (time.perf_counter() - t0) * 1000
        result.frontier_model = self._model
        result.provider = self._provider
        return result

    # ------------------------------------------------------------------
    # Anthropic loop
    # ------------------------------------------------------------------

    def _run_anthropic_loop(self, prompt: str) -> OrchestratorResult:
        """
        Agentic loop for Anthropic (Claude).

        Follows the pattern from examples/claude_agent.py.
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic is required for the Anthropic provider. "
                "Install it with: pip install crasis[agents]"
            )

        client = anthropic.Anthropic(api_key=self._api_key) if self._api_key else anthropic.Anthropic()
        messages: list[dict] = [{"role": "user", "content": prompt}]
        tool_calls_made: list[tuple[str, str, dict]] = []

        for _ in range(self._max_iterations):
            response = client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=self._system_prompt,
                tools=self._toolkit.anthropic_tools(),
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                final_text = next(
                    (block.text for block in response.content if hasattr(block, "text")),
                    "",
                )
                return OrchestratorResult(response=final_text, tool_calls=tool_calls_made)

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    result = self._toolkit.handle_tool_use(block)
                    tool_results.append(result)
                    import json
                    tool_calls_made.append(
                        (block.name, block.input.get("text", ""), json.loads(result["content"]))
                    )
                messages.append({"role": "user", "content": tool_results})

        return OrchestratorResult(
            response="[max_iterations reached]",
            tool_calls=tool_calls_made,
        )

    # ------------------------------------------------------------------
    # OpenAI loop
    # ------------------------------------------------------------------

    def _run_openai_loop(self, prompt: str) -> OrchestratorResult:
        """
        Agentic loop for OpenAI (or any OpenAI-compatible API).
        """
        try:
            import openai
            import json
        except ImportError:
            raise ImportError(
                "openai is required for the OpenAI provider. "
                "Install it with: pip install crasis[agents]"
            )

        client = openai.OpenAI(api_key=self._api_key) if self._api_key else openai.OpenAI()
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]
        tool_calls_made: list[tuple[str, str, dict]] = []

        for _ in range(self._max_iterations):
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=self._toolkit.openai_tools(),
                tool_choice="auto",
            )

            choice = response.choices[0]
            messages.append(choice.message)

            if choice.finish_reason == "stop":
                return OrchestratorResult(
                    response=choice.message.content or "",
                    tool_calls=tool_calls_made,
                )

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    result_json = self._toolkit.handle_tool_call(tool_call)
                    tool_calls_made.append(
                        (
                            tool_call.function.name,
                            json.loads(tool_call.function.arguments).get("text", ""),
                            json.loads(result_json),
                        )
                    )
                    messages.append(self._toolkit.openai_tool_message(tool_call, result_json))

        return OrchestratorResult(
            response="[max_iterations reached]",
            tool_calls=tool_calls_made,
        )

    # ------------------------------------------------------------------
    # Gemini loop
    # ------------------------------------------------------------------

    def _run_gemini_loop(self, prompt: str) -> OrchestratorResult:
        """
        Agentic loop for Google Gemini.
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is required for the Gemini provider. "
                "Install it with: pip install crasis[agents]"
            )

        if self._api_key:
            genai.configure(api_key=self._api_key)

        gemini_tools = self._toolkit.gemini_tools()
        model = genai.GenerativeModel(
            model_name=self._model,
            tools=gemini_tools,
            system_instruction=self._system_prompt,
        )

        chat = model.start_chat()
        tool_calls_made: list[tuple[str, str, dict]] = []

        response = chat.send_message(prompt)

        for _ in range(self._max_iterations):
            # Check if any candidate contains function calls
            function_calls = []
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call.name:
                        function_calls.append(part.function_call)

            if not function_calls:
                # Final text response
                final_text = response.text if hasattr(response, "text") else ""
                return OrchestratorResult(response=final_text, tool_calls=tool_calls_made)

            # Execute all function calls and build responses
            import google.generativeai.types as genai_types

            function_responses = []
            for fc in function_calls:
                result_dict = self._toolkit.handle_gemini_call(fc)
                tool_calls_made.append(
                    (fc.name, dict(fc.args).get("text", ""), result_dict)
                )
                function_responses.append(
                    genai_types.Part.from_function_response(
                        name=fc.name,
                        response=result_dict,
                    )
                )

            response = chat.send_message(function_responses)

        final_text = response.text if hasattr(response, "text") else ""
        return OrchestratorResult(
            response=final_text or "[max_iterations reached]",
            tool_calls=tool_calls_made,
        )
