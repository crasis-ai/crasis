"""
crasis.mcp_server — MCP server that exposes Crasis specialists as local tools.

Runs a stdio-based MCP server. Any MCP client (Claude Desktop, Claude Code,
Cursor, or any custom agent) can call specialists and manage the registry.

Usage:
    crasis mcp --models-dir ~/.crasis/specialists/

Claude Desktop config (~/.config/claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "crasis": {
          "command": "/path/to/crasis",
          "args": ["mcp"],
          "env": {"OPENROUTER_API_KEY": "sk-or-v1-..."}
        }
      }
    }

Tools exposed:
    classify_<name>    — Classify text with a loaded specialist
    list_specialists   — List all loaded specialists
    pull_specialist    — Download a specialist from the registry
    build_specialist   — Run the full build pipeline from a spec YAML string
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


async def run_server(
    models_dir: str | Path,
    api_key: str | None = None,
    data_dir: str | Path | None = None,
) -> None:
    """
    Start the Crasis MCP server over stdio.

    Discovers all specialists in models_dir and serves them as MCP tools.
    Also exposes list_specialists, pull_specialist, and build_specialist.
    Blocks until the client disconnects.

    Args:
        models_dir: Directory containing exported specialist packages.
        api_key: OpenRouter API key for build_specialist tool.
        data_dir: Directory for training data (used by build_specialist).
    """
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp import types
    except ImportError:
        raise ImportError(
            "The 'mcp' package is required to run the Crasis MCP server.\n"
            "Install it with: pip install crasis[mcp]"
        )

    from crasis.tools import CrasisToolkit

    models_dir = Path(models_dir)
    data_dir = Path(data_dir) if data_dir else Path.home() / ".crasis" / "data"

    # Load specialists — tolerate empty directory (pull_specialist can populate it later)
    try:
        toolkit = CrasisToolkit.from_dir(models_dir)
    except FileNotFoundError:
        toolkit = CrasisToolkit({})
        logger.warning("No specialists found in %s — only list/pull/build tools available", models_dir)

    logger.info("MCP server starting with specialists: %s", toolkit.specialists())

    server = Server("crasis")

    # ------------------------------------------------------------------
    # Tool listing
    # ------------------------------------------------------------------

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        """Advertise all loaded specialists plus management tools as MCP tools."""
        tools: list[types.Tool] = []

        # One classify tool per loaded specialist
        for name in toolkit.specialists():
            specialist = toolkit.get_specialist(name)
            classes = " | ".join(specialist.label_names)
            tool_name = f"classify_{name.replace('-', '_')}"
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

        # list_specialists
        tools.append(
            types.Tool(
                name="list_specialists",
                description="List all specialists currently loaded in this MCP server.",
                inputSchema={"type": "object", "properties": {}},
            )
        )

        # pull_specialist
        tools.append(
            types.Tool(
                name="pull_specialist",
                description=(
                    "Download a pre-built specialist from the Crasis registry (GitHub Releases). "
                    "After pulling, restart the MCP server to load the new specialist."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Specialist name to pull (e.g. 'sentiment-gate')",
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Re-download even if already cached",
                        },
                    },
                    "required": ["name"],
                },
            )
        )

        # build_specialist
        tools.append(
            types.Tool(
                name="build_specialist",
                description=(
                    "Run the full Crasis build pipeline (generate → train → export) from a spec YAML string. "
                    "Requires train dependencies: pip install crasis[train]. "
                    "Sends log progress notifications during the build."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "spec_yaml": {
                            "type": "string",
                            "description": "Full YAML content of a crasis spec (not a file path)",
                        },
                        "volume_override": {
                            "type": "integer",
                            "description": "Override training.volume from spec",
                        },
                    },
                    "required": ["spec_yaml"],
                },
            )
        )

        return tools

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        """Route a tool call to the appropriate handler."""

        # --- classify_* ---
        if name.startswith("classify_"):
            return await _handle_classify(toolkit, name, arguments)

        # --- list_specialists ---
        if name == "list_specialists":
            return _handle_list_specialists(toolkit)

        # --- pull_specialist ---
        if name == "pull_specialist":
            return await _handle_pull(models_dir, name, arguments, server, toolkit)

        # --- build_specialist ---
        if name == "build_specialist":
            return await _handle_build(
                models_dir=models_dir,
                data_dir=data_dir,
                api_key=api_key,
                arguments=arguments,
                server=server,
                toolkit=toolkit,
            )

        return [types.TextContent(type="text", text=f"Error: Unknown tool '{name}'")]

    # ------------------------------------------------------------------
    # Start serving
    # ------------------------------------------------------------------

    logger.info(
        "Crasis MCP server ready — %d specialist tools + 3 management tools",
        len(toolkit.specialists()),
    )

    async with stdio_server() as streams:
        await server.run(
            streams[0],
            streams[1],
            server.create_initialization_options(),
        )


# ---------------------------------------------------------------------------
# Handler implementations
# ---------------------------------------------------------------------------


async def _handle_classify(toolkit, tool_name: str, arguments: dict) -> list:
    """Dispatch a classify_* tool call to the correct specialist."""
    from mcp import types

    try:
        text = arguments.get("text", "")
        if not text:
            raise ValueError("'text' argument is required")

        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: toolkit._dispatch(tool_name, arguments),
        )

        parsed = json.loads(result)
        formatted = (
            f"label: {parsed['label']}\n"
            f"confidence: {parsed['confidence']}\n"
            f"latency_ms: {parsed['latency_ms']}"
        )
        return [types.TextContent(type="text", text=formatted)]

    except KeyError as exc:
        return [types.TextContent(type="text", text=f"Error: Unknown tool '{tool_name}'. {exc}")]
    except Exception as exc:
        logger.error("Classify failed: %s", exc, exc_info=True)
        return [types.TextContent(type="text", text=f"Error: {exc}")]


def _handle_list_specialists(toolkit) -> list:
    """Return a JSON list of loaded specialists."""
    from mcp import types

    result = [
        {
            "name": name,
            "labels": toolkit.get_specialist(name).label_names,
            "tool_name": f"classify_{name.replace('-', '_')}",
        }
        for name in toolkit.specialists()
    ]
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_pull(models_dir: Path, _tool_name: str, arguments: dict, server, toolkit) -> list:
    """Download a specialist from the registry and hot-reload it into the toolkit."""
    from mcp import types

    specialist_name = arguments.get("name", "").strip()
    force = bool(arguments.get("force", False))

    if not specialist_name:
        return [types.TextContent(type="text", text="Error: 'name' argument is required")]

    progress_q: queue.SimpleQueue[str] = queue.SimpleQueue()
    result_holder: list = []  # [0] = pkg_dir Path or exception

    def _pull_sync() -> None:
        from crasis.cli import _pull_specialist_sync

        try:
            pkg_dir = _pull_specialist_sync(
                name=specialist_name,
                cache_dir=models_dir,
                force=force,
                progress_callback=progress_q.put,
            )
            result_holder.append(pkg_dir)
        except Exception as exc:
            result_holder.append(exc)
        finally:
            progress_q.put("DONE")

    thread = threading.Thread(target=_pull_sync, daemon=True)
    thread.start()

    while thread.is_alive() or not progress_q.empty():
        try:
            msg = progress_q.get_nowait()
            if msg == "DONE":
                break
            try:
                await server.request_context.session.send_log_message(level="info", data=msg)
            except Exception:
                pass
        except queue.Empty:
            await asyncio.sleep(0.2)

    thread.join()

    outcome = result_holder[0] if result_holder else RuntimeError("Pull produced no result")

    if isinstance(outcome, Exception):
        return [types.TextContent(type="text", text=f"ERROR: {outcome}")]

    # Hot-reload into the live toolkit
    pkg_dir: Path = outcome
    try:
        from crasis.deploy import Specialist

        new_specialist = Specialist.load(pkg_dir)
        toolkit._specialists[new_specialist.name] = new_specialist
        return [
            types.TextContent(
                type="text",
                text=(
                    f"Specialist '{specialist_name}' downloaded and loaded. "
                    f"Tool 'classify_{specialist_name.replace('-', '_')}' is now available."
                ),
            )
        ]
    except Exception as exc:
        logger.warning("Pull succeeded but hot-reload failed: %s", exc)
        return [
            types.TextContent(
                type="text",
                text=(
                    f"Specialist '{specialist_name}' downloaded to {pkg_dir}. "
                    f"Hot-reload failed ({exc}) — restart the MCP server to use it."
                ),
            )
        ]


async def _handle_build(
    models_dir: Path,
    data_dir: Path,
    api_key: str | None,
    arguments: dict,
    server,
    toolkit=None,
) -> list:
    """
    Run the full build pipeline by shelling out to `crasis build`.

    This ensures the MCP build path is identical to the CLI build path —
    same environment, same working directory assumptions, same error surface.
    Progress lines from stderr are forwarded as MCP log notifications.
    """
    from mcp import types

    spec_yaml = arguments.get("spec_yaml", "").strip()
    volume_override = arguments.get("volume_override")

    if not spec_yaml:
        return [types.TextContent(type="text", text="Error: 'spec_yaml' argument is required")]

    if not api_key:
        return [
            types.TextContent(
                type="text",
                text=(
                    "Error: OPENROUTER_API_KEY is required for build_specialist. "
                    "Start the MCP server with --api-key or set the OPENROUTER_API_KEY env var."
                ),
            )
        ]

    # Write spec YAML to a temp file — crasis build requires a file path
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(spec_yaml)
        tmp_spec_path = Path(tmp.name)

    # Parse spec name before launching subprocess so we know where to find
    # the exported package for hot-reload after the build completes
    spec_name: str | None = None
    try:
        from crasis.spec import CrasisSpec

        spec = CrasisSpec.from_yaml(tmp_spec_path)
        spec_name = spec.name
    except Exception as exc:
        tmp_spec_path.unlink(missing_ok=True)
        return [types.TextContent(type="text", text=f"Error parsing spec: {exc}")]

    cmd = [
        "crasis", "build",
        "--spec", str(tmp_spec_path),
        "--api-key", api_key,
        "--model-dir", str(models_dir),
        "--data-dir", str(data_dir),
    ]
    if volume_override is not None:
        # crasis build doesn't have --volume; pass via env so the spec is respected.
        # volume_override is advisory — the spec controls the actual count.
        # Re-write the spec with the overridden volume so the CLI sees it.
        try:
            spec.training.volume = int(volume_override)
            tmp_spec_path.write_text(
                __import__("yaml").dump(spec.model_dump()), encoding="utf-8"
            )
        except Exception:
            pass  # proceed with original spec if rewrite fails

    import os as _os
    env = {**_os.environ, "OPENROUTER_API_KEY": api_key}

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )

        async for raw_line in proc.stdout:
            line = raw_line.decode(errors="replace").rstrip()
            if not line:
                continue
            try:
                await server.request_context.session.send_log_message(level="info", data=line)
            except Exception:
                pass

        await proc.wait()
    finally:
        tmp_spec_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        return [
            types.TextContent(
                type="text",
                text=f"Error: Build failed (exit {proc.returncode}). Check log notifications for details.",
            )
        ]

    # Locate the exported package — crasis build writes to <model-dir>/<name>-onnx/
    pkg_dir = models_dir / f"{spec_name}-onnx"
    if not pkg_dir.exists():
        # Fallback: any subdir containing the onnx file
        candidates = list(models_dir.glob(f"{spec_name}*/*.onnx"))
        if candidates:
            pkg_dir = candidates[0].parent

    # Hot-reload into live toolkit
    if toolkit is not None and pkg_dir.exists():
        try:
            from crasis.deploy import Specialist

            new_specialist = Specialist.load(pkg_dir)
            toolkit._specialists[new_specialist.name] = new_specialist
            return [
                types.TextContent(
                    type="text",
                    text=(
                        f"Build complete and loaded: {pkg_dir}. "
                        f"Tool 'classify_{new_specialist.name.replace('-', '_')}' is now available."
                    ),
                )
            ]
        except Exception as exc:
            logger.warning("Build succeeded but hot-reload failed: %s", exc)
            return [
                types.TextContent(
                    type="text",
                    text=(
                        f"Build complete: {pkg_dir}. "
                        f"Hot-reload failed ({exc}) — restart the MCP server to use it."
                    ),
                )
            ]

    return [types.TextContent(type="text", text=f"Build complete: {pkg_dir}")]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def generate_claude_desktop_config(
    models_dir: str | Path,
    crasis_executable: str = "crasis",
    api_key: str | None = None,
) -> dict:
    """
    Generate a Claude Desktop MCP server configuration block.

    Args:
        models_dir: Absolute path to the specialists directory.
        crasis_executable: Path to the crasis CLI executable.
        api_key: Optional OpenRouter API key to bake into the config env block.

    Returns:
        Dict to merge into claude_desktop_config.json under "mcpServers".
    """
    config: dict = {
        "command": crasis_executable,
        "args": ["mcp", "--models-dir", str(Path(models_dir).absolute())],
    }
    if api_key:
        config["env"] = {"OPENROUTER_API_KEY": api_key}

    return {"crasis": config}
