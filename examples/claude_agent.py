"""
examples/claude_agent.py — Claude using Crasis specialists as local tools.

Claude handles reasoning and conversation. Crasis handles fast local classification.
The specialist is called as a tool — Claude never pays tokens to answer
"is this customer angry?" because a 17MB local model does it in 40ms.

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Usage:
    python examples/claude_agent.py
"""

import json
import anthropic
from crasis.tools import CrasisToolkit


def run_claude_with_crasis(message: str, models_dir: str = "./models") -> str:
    """
    Run a Claude agent that can call local Crasis specialists as tools.

    Claude will automatically call classify_sentiment_gate (or any other
    loaded specialist) when it needs to classify text — running the model
    locally in <100ms rather than burning tokens on a second frontier call.

    Args:
        message: The user message for Claude to process.
        models_dir: Path to directory containing exported specialist packages.

    Returns:
        Claude's final response as a string.
    """
    client = anthropic.Anthropic()
    toolkit = CrasisToolkit.from_dir(models_dir)

    messages = [{"role": "user", "content": message}]

    print(f"\n-> User: {message}")
    print(f"  Crasis tools available: {toolkit.specialists()}\n")

    # Agentic loop — Claude may call tools multiple times before responding
    while True:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            tools=toolkit.anthropic_tools(),
            messages=messages,
        )

        # Append assistant response to history
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Claude is done — extract final text
            final_text = next(
                (block.text for block in response.content if hasattr(block, "text")),
                "",
            )
            print(f"← Claude: {final_text}")
            return final_text

        if response.stop_reason == "tool_use":
            # Claude called one or more Forge tools — handle them all
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                print(f"  Tool call: {block.name}({block.input})")
                result = toolkit.handle_tool_use(block)
                tool_results.append(result)

                # Log the local inference result
                result_data = json.loads(result["content"])
                print(
                    f"  Local result: label={result_data['label']} "
                    f"confidence={result_data['confidence']} "
                    f"latency={result_data['latency_ms']}ms  <- no API call"
                )

            # Feed tool results back to Claude
            messages.append({"role": "user", "content": tool_results})

        else:
            # Unexpected stop reason
            break

    return ""


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    test_cases = [
        "This customer just sent: 'I WANT MY MONEY BACK RIGHT NOW, THIS IS ABSOLUTELY UNACCEPTABLE!!!' — how should I respond?",
        "Handle this support message: 'Hi, the delivery was a bit late. Could you let me know what happened? Thanks'",
        "Triage these two messages and tell me which needs immediate attention:\n1. 'Your service is terrible, I'm cancelling everything'\n2. 'Just checking on my order status, no rush'",
    ]

    for msg in test_cases:
        print("=" * 70)
        run_claude_with_crasis(msg)
        print()
