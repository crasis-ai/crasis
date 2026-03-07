"""
crasis.mcp_server — MCP server that exposes Crasis specialists as local tools.

Runs a stdio-based MCP server. Any MCP client (Claude Desktop, Claude Code,
Cursor, or any custom agent) can call your specialists as local tools.

Usage:
    # Start the server (called by MCP client automatically)
    crasis serve --models ./models

Claude Desktop config (~/.config/claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "crasis": {
          "command": "crasis",
          "args": ["serve", "--models", "/path/to/models"]
        }
      }
    }

The server auto-discovers all specialists in the models directory and
exposes each as a tool named classify_<specialist_name>.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


async def run_server(models_dir: str | Path) -> None:
    """
    Start the Crasis MCP server.

    Discovers all specialists in models_dir and serves them as MCP tools
    over stdio. Blocks until the client disconnects.

    Args:
        models_dir: Directory containing exported specialist packages.
    """
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp import types
    except ImportError:
        raise ImportError(
            "The 'mcp' package is required to run the Crasis MCP server.\n"
            "Install it with: pip install mcp"
        )

    from crasis.tools import CrasisToolkit

    # Load all specialists
    toolkit = CrasisToolkit.from_dir(models_dir)
    specialist_names = toolkit.specialists()
    logger.info("MCP server starting with specialists: %s", specialist_names)

    server = Server("crasis")

    # ------------------------------------------------------------------
    # Tool listing
    # ------------------------------------------------------------------

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        """Advertise all loaded specialists as MCP tools."""
        tools = []
        for name in toolkit.specialists():
            tool_name = f"classify_{name.replace('-', '_')}"
            # Load specialist to get label names
            specialist = toolkit._specialists[name]
            classes = " | ".join(specialist.label_names)

            tools.append(
                types.Tool(
                    name=tool_name,
                    description=(
                        f"Classify text using the '{name}' Crasis specialist. "
                        f"Returns: {classes}. "
                        f"Runs locally in <100ms. No cloud API calls."
                    ),
                    inputSchema={
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
            )

        return tools

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent]:
        """Execute a specialist classification and return the result."""
        try:
            text = arguments.get("text", "")
            if not text:
                raise ValueError("'text' argument is required")

            # Dispatch — run in thread pool to avoid blocking the event loop
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: toolkit._dispatch(name, arguments),
            )

            parsed = json.loads(result)
            formatted = (
                f"label: {parsed['label']}\n"
                f"confidence: {parsed['confidence']}\n"
                f"latency_ms: {parsed['latency_ms']}"
            )

            return [types.TextContent(type="text", text=formatted)]

        except KeyError as exc:
            return [
                types.TextContent(
                    type="text",
                    text=f"Error: Unknown tool '{name}'. {exc}",
                )
            ]
        except Exception as exc:
            logger.error("Tool call failed: %s", exc, exc_info=True)
            return [
                types.TextContent(
                    type="text",
                    text=f"Error: {exc}",
                )
            ]

    # ------------------------------------------------------------------
    # Start serving
    # ------------------------------------------------------------------

    logger.info("Crasis MCP server ready — %d tools available", len(specialist_names))

    async with stdio_server() as streams:
        await server.run(
            streams[0],
            streams[1],
            server.create_initialization_options(),
        )


def generate_claude_desktop_config(
    models_dir: str | Path,
    crasis_executable: str = "crasis",
) -> dict:
    """
    Generate a Claude Desktop MCP server configuration block.

    Args:
        models_dir: Absolute path to the models directory.
        crasis_executable: Path to the crasis CLI executable.

    Returns:
        Dict to merge into claude_desktop_config.json under "mcpServers".
    """
    return {
        "crasis": {
            "command": crasis_executable,
            "args": ["serve", "--models", str(Path(models_dir).absolute())],
        }
    }
