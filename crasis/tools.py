"""
crasis.tools — OpenAI / Anthropic / Gemini tool definitions for Crasis specialists.

Converts loaded specialists into function-calling schemas that any frontier model
can invoke. The computation happens locally. The frontier model only sees the result.

Usage with OpenAI:
    from crasis.tools import CrasisToolkit
    from openai import OpenAI

    toolkit = CrasisToolkit.from_dir("./models")
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Is this customer angry? 'I WANT A REFUND NOW'"}],
        tools=toolkit.openai_tools(),
        tool_choice="auto",
    )

    result = toolkit.handle_tool_call(response.choices[0].message.tool_calls[0])

Usage with Anthropic:
    import anthropic
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        tools=toolkit.anthropic_tools(),
        messages=[{"role": "user", "content": "..."}],
    )

    result = toolkit.handle_tool_use(response.content[0])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from crasis.deploy import Specialist

logger = logging.getLogger(__name__)


class CrasisToolkit:
    """
    A collection of loaded Crasis specialists exposed as LLM tool definitions.

    Each specialist becomes a tool named `classify_<specialist_name>` that
    accepts a single `text` argument and returns a classification result.

    The toolkit handles tool dispatch — pass it a raw tool call from any
    supported frontier model and it routes to the correct specialist.
    """

    def __init__(self, specialists: dict[str, Specialist]) -> None:
        """
        Args:
            specialists: Dict mapping specialist name → loaded Specialist instance.
        """
        self._specialists = specialists
        logger.info(
            "CrasisToolkit loaded %d specialists: %s",
            len(specialists),
            list(specialists.keys()),
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_dir(cls, models_dir: str | Path) -> "CrasisToolkit":
        """
        Load all specialists found in a models directory.

        Looks for any subdirectory containing a .onnx file.

        Args:
            models_dir: Root directory containing specialist packages.

        Returns:
            A CrasisToolkit with all discovered specialists loaded.
        """
        models_dir = Path(models_dir)
        specialists: dict[str, Specialist] = {}

        for candidate in sorted(models_dir.iterdir()):
            if not candidate.is_dir():
                continue
            onnx_files = list(candidate.glob("*.onnx"))
            if not onnx_files:
                continue
            try:
                s = Specialist.load(candidate)
                specialists[s.name] = s
                logger.info("Loaded specialist: %s", s.name)
            except Exception as exc:
                logger.warning("Failed to load specialist from %s: %s", candidate, exc)

        if not specialists:
            raise FileNotFoundError(
                f"No specialist packages found in {models_dir}. "
                "Run `crasis export` to create deployable packages first."
            )

        return cls(specialists)

    @classmethod
    def from_specialists(cls, *specialists: Specialist) -> "CrasisToolkit":
        """
        Create a toolkit from explicitly provided Specialist instances.

        Args:
            *specialists: One or more loaded Specialist instances.
        """
        return cls({s.name: s for s in specialists})

    # ------------------------------------------------------------------
    # OpenAI / compatible (OpenAI, Groq, Together, OpenRouter)
    # ------------------------------------------------------------------

    def openai_tools(self) -> list[dict]:
        """
        Return OpenAI-compatible tool definitions for all loaded specialists.

        Compatible with: OpenAI, Groq, Together AI, OpenRouter,
        any API that follows the OpenAI function calling spec.

        Returns:
            List of tool definition dicts ready to pass as `tools=` parameter.
        """
        return [self._openai_tool_def(s) for s in self._specialists.values()]

    def handle_tool_call(self, tool_call) -> str:
        """
        Execute an OpenAI tool call and return a JSON string result.

        Args:
            tool_call: A ToolCall object from an OpenAI chat completion response.

        Returns:
            JSON string — the classification result from the specialist.

        Raises:
            KeyError: If the tool name doesn't match any loaded specialist.
        """
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        return self._dispatch(name, args)

    def openai_tool_message(self, tool_call, result: str) -> dict:
        """
        Build the tool result message to append to an OpenAI conversation.

        Args:
            tool_call: The original ToolCall object.
            result: JSON string returned by handle_tool_call.

        Returns:
            A message dict with role="tool" ready to append to messages list.
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        }

    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------

    def anthropic_tools(self) -> list[dict]:
        """
        Return Anthropic-compatible tool definitions for all loaded specialists.

        Returns:
            List of tool definition dicts ready to pass as `tools=` parameter
            to the Anthropic messages API.
        """
        return [self._anthropic_tool_def(s) for s in self._specialists.values()]

    def handle_tool_use(self, tool_use_block) -> dict:
        """
        Execute an Anthropic tool_use content block and return a result dict.

        Args:
            tool_use_block: A ToolUseBlock from an Anthropic message response.

        Returns:
            Dict suitable for use as a tool_result content block.
        """
        name = tool_use_block.name
        args = tool_use_block.input
        result_json = self._dispatch(name, args)

        return {
            "type": "tool_result",
            "tool_use_id": tool_use_block.id,
            "content": result_json,
        }

    # ------------------------------------------------------------------
    # Gemini
    # ------------------------------------------------------------------

    def gemini_tools(self) -> list:
        """
        Return Gemini-compatible tool definitions.

        Requires: google-generativeai

        Returns:
            List of FunctionDeclaration objects for use with Gemini.
        """
        try:
            from google.generativeai.types import FunctionDeclaration, Tool
        except ImportError:
            raise ImportError(
                "google-generativeai is required for Gemini tools. "
                "pip install google-generativeai"
            )

        declarations = [
            FunctionDeclaration(
                name=self._tool_name(s),
                description=self._tool_description(s),
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to classify",
                        }
                    },
                    "required": ["text"],
                },
            )
            for s in self._specialists.values()
        ]

        return [Tool(function_declarations=declarations)]

    def handle_gemini_call(self, function_call) -> dict:
        """
        Execute a Gemini function call.

        Args:
            function_call: A FunctionCall from a Gemini candidate.

        Returns:
            Dict with the classification result.
        """
        name = function_call.name
        args = dict(function_call.args)
        result_json = self._dispatch(name, args)
        return json.loads(result_json)

    # ------------------------------------------------------------------
    # Direct dispatch (framework-agnostic)
    # ------------------------------------------------------------------

    def classify(self, specialist_name: str, text: str) -> dict:
        """
        Directly classify text with a named specialist.

        This is the framework-agnostic path — use when you're not going
        through a frontier model's tool-calling interface.

        Args:
            specialist_name: The specialist's name (e.g. "sentiment-gate").
            text: Text to classify.

        Returns:
            Classification result dict.

        Raises:
            KeyError: If specialist_name is not in this toolkit.
        """
        if specialist_name not in self._specialists:
            raise KeyError(
                f"Specialist '{specialist_name}' not found. "
                f"Available: {list(self._specialists.keys())}"
            )
        return self._specialists[specialist_name].classify(text)

    def get_specialist(self, name: str) -> "Specialist":
        """
        Return a loaded Specialist by name.

        Args:
            name: Specialist name (e.g. "sentiment-gate").

        Returns:
            Loaded Specialist instance.

        Raises:
            KeyError: If name is not in this toolkit.
        """
        if name not in self._specialists:
            raise KeyError(
                f"Specialist '{name}' not found. Available: {list(self._specialists.keys())}"
            )
        return self._specialists[name]

    def specialists(self) -> list[str]:
        """Return list of loaded specialist names."""
        return list(self._specialists.keys())

    def __repr__(self) -> str:
        return f"CrasisToolkit(specialists={self.specialists()})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tool_name(s: Specialist) -> str:
        """Convert specialist name to a valid function name."""
        return f"classify_{s.name.replace('-', '_')}"

    @staticmethod
    def _tool_description(s: Specialist) -> str:
        """Build a tool description from specialist metadata."""
        classes = " | ".join(s.label_names)
        return (
            f"Classify text using the '{s.name}' local specialist model. "
            f"Returns one of: [{classes}] with a confidence score. "
            f"Runs locally in <100ms. No API calls made."
        )

    def _openai_tool_def(self, s: Specialist) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self._tool_name(s),
                "description": self._tool_description(s),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to classify",
                        }
                    },
                    "required": ["text"],
                },
            },
        }

    def _anthropic_tool_def(self, s: Specialist) -> dict:
        return {
            "name": self._tool_name(s),
            "description": self._tool_description(s),
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to classify",
                    }
                },
                "required": ["text"],
            },
        }

    def _dispatch(self, tool_name: str, args: dict) -> str:
        """Route a tool call to the correct specialist and return JSON result."""
        # tool_name is like "classify_sentiment_gate"
        # specialist name is like "sentiment-gate"
        specialist_name = tool_name.removeprefix("classify_").replace("_", "-")

        if specialist_name not in self._specialists:
            raise KeyError(
                f"No specialist found for tool '{tool_name}'. "
                f"Available: {list(self._specialists.keys())}"
            )

        text = args.get("text", "")
        result = self._specialists[specialist_name].classify(text)
        return json.dumps(result)


# ---------------------------------------------------------------------------
# Backwards-compatible alias
# ---------------------------------------------------------------------------

ForgeToolkit = CrasisToolkit
