# Agentic Workflows — MCP Server & Orchestrator

Crasis exposes two ways to slot specialist models into agentic workflows:

1. **MCP server** (`crasis mcp`) — run a stdio MCP server that any MCP client (Claude Desktop, Claude Code, Cursor) can connect to and call specialists as tools.
2. **`CrasisOrchestrator`** — a Python class that wires the multi-turn tool-calling loop for OpenAI, Anthropic, and Gemini so you don't have to do it yourself.

In both cases: the frontier model handles reasoning. The specialist handles classification locally in <100ms with no API calls.

---

## Install

```bash
# MCP server only
pip install "crasis[mcp]"

# Orchestrator only
pip install "crasis[agents]"

# Everything
pip install "crasis[all]"
```

---

## MCP Server

### Start the server

```bash
crasis mcp
```

By default the server looks for specialists in `~/.crasis/specialists/`. Override with `--models-dir`:

```bash
crasis mcp --models-dir ./models --api-key sk-or-v1-...
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--models-dir` | `~/.crasis/specialists/` | Directory containing specialist packages |
| `--api-key` | `$OPENROUTER_API_KEY` | Only needed if you plan to call `build_specialist` |
| `--data-dir` | `~/.crasis/data/` | Training data directory for `build_specialist` |

### Connect Claude Desktop

Add to `~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "crasis": {
      "command": "/home/you/venv/bin/crasis",
      "args": ["mcp"],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-v1-..."
      }
    }
  }
}
```

Or generate the config block from Python:

```python
from crasis.mcp_server import generate_claude_desktop_config

cfg = generate_claude_desktop_config(
    models_dir="~/.crasis/specialists",
    crasis_executable="/home/you/venv/bin/crasis",
    api_key="sk-or-v1-...",
)
# merge cfg into your claude_desktop_config.json under "mcpServers"
```

### Tools exposed

The server exposes four tool types:

**`classify_<name>`** — one per loaded specialist.

```
classify_sentiment_gate(text: string) → label, confidence, latency_ms
classify_spam_filter(text: string) → label, confidence, latency_ms
...
```

**`list_specialists`** — list all loaded specialists with their labels and tool names.

```json
[
  {
    "name": "sentiment-gate",
    "labels": ["negative", "positive"],
    "tool_name": "classify_sentiment_gate"
  }
]
```

**`pull_specialist`** — download a specialist from the registry.

```
pull_specialist(name: string, force?: bool)
```

After pulling, restart the MCP server to load the new specialist.

**`build_specialist`** — run the full build pipeline from a spec YAML string.

```
build_specialist(spec_yaml: string, volume_override?: int)
```

`spec_yaml` is the YAML content (not a file path) because MCP clients can't reliably share filesystem paths with the server. The server writes it to a temp file, parses it, and deletes it. Requires `pip install crasis[train]` and `--api-key`.

The server streams log notifications during `pull_specialist` and `build_specialist` so you see progress in your MCP client.

**Note:** `pull_specialist` and `build_specialist` add specialists to disk but do NOT hot-reload them into a running server. Restart the server after either operation.

---

## CrasisOrchestrator

The orchestrator wraps the multi-turn agentic loop for three providers. You give it a prompt; it handles everything until the frontier model produces a final text response.

### Quick start

```python
from crasis import CrasisToolkit, CrasisOrchestrator

toolkit = CrasisToolkit.from_dir("~/.crasis/specialists")

orch = CrasisOrchestrator(
    toolkit=toolkit,
    provider="anthropic",        # "openai" | "anthropic" | "gemini"
    model="claude-opus-4-5",
)

result = orch.run("Is this customer angry? 'I WANT A REFUND RIGHT NOW'")

print(result.response)       # Claude's final answer
print(result.tool_calls)     # [(tool_name, input_text, result_dict), ...]
print(result.total_latency_ms)
```

### OrchestratorResult fields

| Field | Type | Description |
|---|---|---|
| `response` | `str` | Final text response from the frontier model |
| `tool_calls` | `list[tuple[str, str, dict]]` | `(tool_name, input_text, result)` for each specialist called |
| `total_latency_ms` | `float` | Wall-clock time for the entire run |
| `frontier_model` | `str` | Model ID used |
| `provider` | `str` | Provider name |

### Anthropic

```python
from crasis import CrasisToolkit, CrasisOrchestrator

toolkit = CrasisToolkit.from_dir("~/.crasis/specialists")

orch = CrasisOrchestrator(
    toolkit=toolkit,
    provider="anthropic",
    model="claude-opus-4-5",
    api_key="sk-ant-...",           # or set ANTHROPIC_API_KEY env var
    max_iterations=10,
    system_prompt="You are a customer support triage agent.",
)

result = orch.run(
    "Triage these messages and tell me which needs immediate attention:\n"
    "1. 'I WANT A REFUND NOW'\n"
    "2. 'Just checking on my order, no rush'"
)

for tool_name, input_text, tool_result in result.tool_calls:
    print(f"{tool_name}({input_text!r}) -> {tool_result['label']} ({tool_result['confidence']:.2f})")
```

### OpenAI

```python
from crasis import CrasisToolkit, CrasisOrchestrator

toolkit = CrasisToolkit.from_dir("~/.crasis/specialists")

orch = CrasisOrchestrator(
    toolkit=toolkit,
    provider="openai",
    model="gpt-4o",
    api_key="sk-...",              # or set OPENAI_API_KEY env var
)

result = orch.run("Does this email contain pricing information? 'Our enterprise plan starts at $500/mo'")
print(result.response)
```

### Gemini

```python
from crasis import CrasisToolkit, CrasisOrchestrator

toolkit = CrasisToolkit.from_dir("~/.crasis/specialists")

orch = CrasisOrchestrator(
    toolkit=toolkit,
    provider="gemini",
    model="gemini-pro",
    api_key="AI...",               # or configure with genai.configure()
)

result = orch.run("Route this support ticket: 'My invoice has a wrong charge'")
print(result.response)
```

### CrasisToolkit.from_specialists()

If you only want to expose specific specialists rather than an entire directory:

```python
from crasis import Specialist
from crasis.tools import CrasisToolkit
from crasis.orchestrator import CrasisOrchestrator

sentiment = Specialist.load("~/.crasis/specialists/sentiment-gate")
spam = Specialist.load("~/.crasis/specialists/spam-filter")

toolkit = CrasisToolkit.from_specialists(sentiment, spam)
orch = CrasisOrchestrator(toolkit=toolkit, provider="anthropic", model="claude-opus-4-5")
```

### Using CrasisToolkit directly (no orchestrator)

If you want to wire the loop yourself — or use a provider the orchestrator doesn't support — `CrasisToolkit` gives you the tool schemas and dispatch methods directly:

```python
from crasis.tools import CrasisToolkit

toolkit = CrasisToolkit.from_dir("~/.crasis/specialists")

# OpenAI / OpenAI-compatible (Groq, Together, OpenRouter)
toolkit.openai_tools()                  # pass as tools= to chat.completions.create
toolkit.handle_tool_call(tool_call)     # returns JSON string
toolkit.openai_tool_message(tc, res)    # builds the tool result message

# Anthropic
toolkit.anthropic_tools()              # pass as tools= to messages.create
toolkit.handle_tool_use(block)          # returns tool_result dict

# Gemini
toolkit.gemini_tools()                  # pass to GenerativeModel
toolkit.handle_gemini_call(fc)          # returns result dict

# Framework-agnostic
toolkit.classify("sentiment-gate", "I want a refund")
toolkit.get_specialist("sentiment-gate")   # returns Specialist instance
toolkit.specialists()                      # returns list of loaded names
```

---

## Architecture

```
User prompt
    │
    ▼
Frontier model (Anthropic / OpenAI / Gemini)
    │  tool_call: classify_sentiment_gate("I WANT A REFUND")
    ▼
CrasisOrchestrator / MCP server
    │  dispatch to local specialist
    ▼
Specialist.classify()  ← ONNX Runtime, <100ms, no network
    │  {"label": "positive", "confidence": 0.97, "latency_ms": 41}
    ▼
Frontier model (sees result, continues reasoning)
    │
    ▼
Final response
```

The frontier model is the orchestrator. The specialist is the instrument. No classification tokens are burned on yes/no questions.
