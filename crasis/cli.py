"""
crasis.cli — Command-line interface.

Commands:
  crasis generate  — Generate synthetic training data
  crasis train     — Train a specialist model
  crasis eval      — Evaluate a trained model
  crasis export    — Export to ONNX
  crasis classify  — Run inference on text input(s)
  crasis build     — Full pipeline: generate → train → eval → export
  crasis pull      — Download a pre-built specialist from the registry
  crasis mcp       — Start the Crasis MCP server over stdio
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()


def _require_train_deps() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        rprint(
            "\n[bold red]Train dependencies are not installed.[/bold red]\n"
            r"  Run: [bold]pip install crasis\[train][/bold]" + "\n"
        )
        sys.exit(1)

# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """Crasis — Train once. Run forever. Pay nothing."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# crasis generate
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--spec", "-s", required=True, type=click.Path(exists=True), help="Path to spec.yaml")
@click.option("--count", "-n", type=int, default=None, help="Override sample count from spec")
@click.option("--output", "-o", default="./data", show_default=True, help="Output directory")
@click.option("--api-key", envvar="OPENROUTER_API_KEY", required=True, help="OpenRouter API key")
@click.option("--no-resume", is_flag=True, default=False, help="Start over even if partial data exists")
def generate(spec: str, count: int | None, output: str, api_key: str, no_resume: bool) -> None:
    """Generate synthetic training data for a specialist."""
    _require_train_deps()
    from crasis.spec import CrasisSpec
    from crasis.factory import generate as _generate

    crasis_spec = CrasisSpec.from_yaml(spec)

    rprint("\n[bold cyan]Crasis — Data Generation[/bold cyan]")
    rprint(f"  Specialist : [bold]{crasis_spec.name}[/bold]")
    rprint(f"  Samples    : [bold]{count or crasis_spec.training.volume}[/bold]")
    rprint("  Strategy   : enforce_distillable_text=true via OpenRouter\n")

    out_path = _generate(
        spec=crasis_spec,
        output_dir=output,
        api_key=api_key,
        count=count,
        resume=not no_resume,
    )

    rprint(f"\n[bold green]✓ Data written to {out_path}[/bold green]")


# ---------------------------------------------------------------------------
# crasis train
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--spec", "-s", required=True, type=click.Path(exists=True), help="Path to spec.yaml")
@click.option("--data", "-d", required=True, type=click.Path(exists=True), help="Path to train.jsonl")
@click.option("--output", "-o", default="./models", show_default=True, help="Output directory")
@click.option("--device", default=None, help="Force device: cpu | cuda | mps")
def train(spec: str, data: str, output: str, device: str | None) -> None:
    """Train a specialist model from generated data."""
    _require_train_deps()
    from crasis.spec import CrasisSpec
    from crasis.train import train as _train

    crasis_spec = CrasisSpec.from_yaml(spec)

    rprint("\n[bold cyan]Crasis — Training[/bold cyan]")
    rprint(f"  Specialist : [bold]{crasis_spec.name}[/bold]")
    rprint(f"  Data       : {data}")
    rprint(f"  Output     : {output}\n")

    result = _train(
        spec=crasis_spec,
        data_path=data,
        output_dir=output,
        device=device,
    )

    _print_train_result(result)


def _print_train_result(result) -> None:
    table = Table(title="Training Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    table.add_row("Base model", result.base_model)
    table.add_row("Train samples", str(result.num_train_samples))
    table.add_row("Eval samples", str(result.num_eval_samples))
    table.add_row("Accuracy", f"{result.eval_accuracy:.4f}")
    table.add_row("F1 (macro)", f"{result.eval_f1:.4f}")
    table.add_row("Duration", f"{result.training_duration_s:.1f}s")
    table.add_row(
        "Quality gate",
        "[green]PASSED ✓[/green]" if result.passed_quality_gate else "[red]FAILED ✗[/red]"
    )
    table.add_row("Model path", str(result.model_path))
    console.print(table)


# ---------------------------------------------------------------------------
# crasis eval
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--spec", "-s", required=True, type=click.Path(exists=True), help="Path to spec.yaml")
@click.option("--model", "-m", required=True, type=click.Path(exists=True), help="Path to model directory")
@click.option("--eval-data", "-e", type=click.Path(exists=True), default=None, help="Path to eval.jsonl")
@click.option(
    "--holdout",
    type=click.Path(exists=True),
    default=None,
    help="Path to hand-authored holdout JSONL for real-world accuracy check",
)
def eval(spec: str, model: str, eval_data: str | None, holdout: str | None) -> None:
    """Evaluate a trained model against spec quality gates.

    Use --holdout to run a second pass on real-world examples and surface
    any gap between synthetic training accuracy and real-world performance.

    Example:

        crasis eval -s specialists/sentiment-gate/spec.yaml \\
            -m ./models/sentiment-gate-onnx \\
            --holdout tests/fixtures/sentiment-gate.jsonl
    """
    _require_train_deps()
    from crasis.spec import CrasisSpec
    from crasis.eval import evaluate as _evaluate, eval_on_holdout as _eval_holdout

    crasis_spec = CrasisSpec.from_yaml(spec)

    try:
        result = _evaluate(
            spec=crasis_spec,
            model_path=model,
            eval_data_path=eval_data,
        )
        gate_color = "green" if result.passed_quality_gate else "red"
        gate_label = "PASSED" if result.passed_quality_gate else "FAILED"
        rprint(f"\n[bold {gate_color}]{'✓' if result.passed_quality_gate else '✗'} Quality gate {gate_label}[/bold {gate_color}]")
        rprint(f"  Accuracy (synthetic) : {result.accuracy:.4f}")
        rprint(f"  F1 macro (synthetic) : {result.f1_macro:.4f}")
        rprint(f"  Samples              : {result.num_samples}")
        if result.classification_report and not result.classification_report.startswith("(cached"):
            rprint(f"\n{result.classification_report}")
    except Exception as exc:
        rprint(f"\n[bold red]✗ {exc}[/bold red]")
        sys.exit(1)

    if holdout:
        from crasis.deploy import Specialist

        rprint("\n[bold cyan]Holdout Evaluation[/bold cyan]")
        rprint(f"  File : {holdout}\n")

        specialist = Specialist.load(model)
        holdout_result = _eval_holdout(
            specialist=specialist,
            holdout_path=holdout,
            spec=crasis_spec,
            synthetic_accuracy=result.accuracy,
        )

        gap = result.accuracy - holdout_result.accuracy
        gap_color = "red" if holdout_result.gap_flagged else "green"

        rprint(f"  Accuracy (holdout)   : {holdout_result.accuracy:.4f}")
        rprint(f"  F1 macro (holdout)   : {holdout_result.f1_macro:.4f}")
        rprint(f"  Samples              : {holdout_result.num_samples}")
        rprint(f"  Synthetic-real gap   : [{gap_color}]{gap:+.4f}[/{gap_color}]")
        if holdout_result.gap_flagged:
            rprint(
                "\n[bold yellow]Warning: holdout accuracy is >5pp below synthetic accuracy.[/bold yellow]"
                "\n  The model may be memorizing synthetic distribution patterns."
                "\n  Consider: more diverse training data, hybrid strategy, or real_data strategy."
            )
        rprint(f"\n{holdout_result.classification_report}")


# ---------------------------------------------------------------------------
# crasis export
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--spec", "-s", required=True, type=click.Path(exists=True), help="Path to spec.yaml")
@click.option("--model", "-m", required=True, type=click.Path(exists=True), help="Path to model directory")
@click.option("--output", "-o", default="./models", show_default=True, help="Output directory")
def export(spec: str, model: str, output: str) -> None:
    """Export a trained model to ONNX."""
    _require_train_deps()
    from crasis.spec import CrasisSpec
    from crasis.export import export as _export

    crasis_spec = CrasisSpec.from_yaml(spec)

    rprint("\n[bold cyan]Crasis — ONNX Export[/bold cyan]")
    rprint(f"  Specialist : [bold]{crasis_spec.name}[/bold]")

    result = _export(
        spec=crasis_spec,
        model_path=model,
        output_dir=output,
    )

    size_color = "green" if result.within_size_constraint else "yellow"
    rprint(f"\n  ONNX path  : {result.onnx_path}")
    rprint(f"  Model size : [{size_color}]{result.model_size_mb:.1f}MB[/{size_color}]")
    rprint("\n[bold green]✓ Export complete[/bold green]")


# ---------------------------------------------------------------------------
# crasis classify
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True), help="Path to specialist package")
@click.argument("texts", nargs=-1)
@click.option("--file", "-f", type=click.Path(exists=True), default=None, help="File with one input per line")
def classify(model: str, texts: tuple[str, ...], file: str | None) -> None:
    """
    Classify text with a deployed specialist.

    Examples:

        crasis classify --model ./models/sentiment-gate "I want a refund"

        echo "This is terrible" | crasis classify --model ./models/sentiment-gate

        crasis classify --model ./models/sentiment-gate --file inputs.txt
    """
    from crasis.deploy import Specialist

    specialist = Specialist.load(model)

    # Collect inputs
    inputs: list[str] = list(texts)

    if file:
        with open(file, encoding="utf-8") as fh:
            inputs.extend(line.strip() for line in fh if line.strip())

    if not inputs and not sys.stdin.isatty():
        inputs = [line.strip() for line in sys.stdin if line.strip()]

    if not inputs:
        rprint("[yellow]No input text provided. Pass text as arguments or use --file.[/yellow]")
        sys.exit(1)

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Text", max_width=60, overflow="fold")
    table.add_column("Label", style="bold")
    table.add_column("Confidence")
    table.add_column("Latency")

    for text in inputs:
        result = specialist.classify(text)
        conf_color = "green" if result["confidence"] >= 0.85 else "yellow"
        table.add_row(
            text[:80],
            result["label"],
            f"[{conf_color}]{result['confidence']:.3f}[/{conf_color}]",
            f"{result['latency_ms']:.1f}ms",
        )

    console.print(f"\n[bold]Specialist:[/bold] {specialist.name}")
    console.print(table)


# ---------------------------------------------------------------------------
# crasis build (full pipeline)
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--spec", "-s", required=True, type=click.Path(exists=True), help="Path to spec.yaml")
@click.option("--api-key", envvar="OPENROUTER_API_KEY", required=True, help="OpenRouter API key")
@click.option("--data-dir", default="./data", show_default=True, help="Training data directory")
@click.option("--model-dir", default="./models", show_default=True, help="Model output directory")
@click.option("--device", default=None, help="Force device: cpu | cuda | mps")
def build(spec: str, api_key: str, data_dir: str, model_dir: str, device: str | None) -> None:
    """
    Full pipeline: generate → train → export.

    This is the one-command path from spec to deployable ONNX.
    """
    _require_train_deps()
    from crasis.spec import CrasisSpec
    from crasis.factory import generate as _generate
    from crasis.train import train as _train
    from crasis.export import export as _export

    crasis_spec = CrasisSpec.from_yaml(spec)

    rprint("\n[bold cyan]Crasis — Full Build Pipeline[/bold cyan]")
    rprint(f"  Specialist : [bold]{crasis_spec.name}[/bold]")
    rprint(f"  Samples    : {crasis_spec.training.volume}")
    rprint(f"  Target     : ≤{crasis_spec.constraints.max_model_size_mb}MB ONNX\n")

    # Step 1: Generate
    rprint("[bold]Step 1/3 — Generating training data...[/bold]")
    data_path = _generate(
        spec=crasis_spec,
        output_dir=data_dir,
        api_key=api_key,
    )

    # Step 2: Train
    rprint("\n[bold]Step 2/3 — Training specialist...[/bold]")
    train_result = _train(
        spec=crasis_spec,
        data_path=data_path,
        output_dir=model_dir,
        device=device,
    )

    if not train_result.passed_quality_gate:
        rprint("[bold red]✗ Quality gate failed. Aborting export.[/bold red]")
        rprint("  Try: increasing training volume, adjusting quality thresholds, or re-running generate.")
        sys.exit(1)

    # Step 3: Export
    rprint("\n[bold]Step 3/3 — Exporting to ONNX...[/bold]")
    export_result = _export(
        spec=crasis_spec,
        model_path=train_result.model_path,
        output_dir=model_dir,
    )

    rprint("\n[bold green]✓ Build complete![/bold green]")
    rprint(f"  Model : {export_result.onnx_path}")
    rprint(f"  Size  : {export_result.model_size_mb:.1f}MB")
    rprint(f"  Run   : crasis classify --model {export_result.package_dir} \"your text here\"")


# ---------------------------------------------------------------------------
# crasis mix
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--spec", "-s", required=True, type=click.Path(exists=True), help="Path to spec.yaml")
@click.option("--real-data", "-r", required=True, type=click.Path(exists=True), help="Path to real-world JSONL examples")
@click.option("--synthetic-data", "-d", default=None, type=click.Path(exists=True), help="Path to synthetic train.jsonl (auto-discovered if omitted)")
@click.option("--output", "-o", default="./models", show_default=True, help="Root output directory")
@click.option("--real-weight", "-w", default=3, show_default=True, type=int, help="Oversample real examples N times relative to synthetic")
@click.option("--device", default=None, help="Force device: cpu | cuda | mps")
def mix(
    spec: str,
    real_data: str,
    synthetic_data: str | None,
    output: str,
    real_weight: int,
    device: str | None,
) -> None:
    """
    Retrain a specialist by mixing real-world examples with synthetic training data.

    Real examples are oversampled (--real-weight) to ensure they influence the
    model despite being fewer in number than the synthetic set. The result is
    exported to a timestamped ONNX package directory.

    Example:

        crasis mix -s specialists/spam-filter/spec.yaml -r ./my-spam.jsonl

        crasis mix -s specialists/spam-filter/spec.yaml \\
            -r ./my-spam.jsonl \\
            --synthetic-data ./data/spam-filter/train.jsonl \\
            --real-weight 5
    """
    _require_train_deps()

    import json
    import random
    import tempfile
    from datetime import datetime
    from pathlib import Path as P

    from crasis.spec import CrasisSpec
    from crasis.train import train as _train
    from crasis.export import export as _export

    crasis_spec = CrasisSpec.from_yaml(spec)
    valid_labels = set(crasis_spec.label_names)

    rprint(f"\n[bold cyan]Crasis — Mix & Retrain[/bold cyan]")
    rprint(f"  Specialist  : [bold]{crasis_spec.name}[/bold]")
    rprint(f"  Valid labels: {sorted(valid_labels)}")
    rprint(f"  Real weight : {real_weight}x\n")

    # ------------------------------------------------------------------
    # Load and validate real data
    # ------------------------------------------------------------------
    real_rows: list[dict] = []
    real_errors: list[str] = []

    with open(real_data, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                real_errors.append(f"  Line {lineno}: invalid JSON — {exc}")
                continue
            if "text" not in row:
                real_errors.append(f"  Line {lineno}: missing 'text' field")
                continue
            if "label" not in row:
                real_errors.append(f"  Line {lineno}: missing 'label' field")
                continue
            if row["label"] not in valid_labels:
                real_errors.append(
                    f"  Line {lineno}: unknown label '{row['label']}' "
                    f"(valid: {sorted(valid_labels)})"
                )
                continue
            real_rows.append({"text": row["text"], "label": row["label"]})

    if real_errors:
        rprint("[bold red]Validation errors in real data:[/bold red]")
        for err in real_errors[:20]:
            rprint(err)
        if len(real_errors) > 20:
            rprint(f"  ... and {len(real_errors) - 20} more")
        sys.exit(1)

    if not real_rows:
        rprint("[bold red]No valid rows found in real data file.[/bold red]")
        sys.exit(1)

    # Per-label counts and low-count warnings
    label_counts: dict[str, int] = {}
    for row in real_rows:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1

    for label in valid_labels:
        count = label_counts.get(label, 0)
        if count == 0:
            rprint(f"[yellow]Warning: no real examples for label '{label}'[/yellow]")
        elif count < 10:
            rprint(f"[yellow]Warning: only {count} real examples for label '{label}' — consider collecting more[/yellow]")

    rprint(f"  Real examples loaded: {len(real_rows)}")
    for label, count in sorted(label_counts.items()):
        rprint(f"    {label}: {count}")

    # ------------------------------------------------------------------
    # Locate synthetic data
    # ------------------------------------------------------------------
    synthetic_rows: list[dict] = []

    synth_path: P | None = None
    if synthetic_data:
        synth_path = P(synthetic_data)
    else:
        candidate = P("./data") / crasis_spec.name / "train.jsonl"
        if candidate.exists():
            synth_path = candidate
            rprint(f"\n  Auto-discovered synthetic data: {synth_path}")

    if synth_path:
        with open(synth_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        synthetic_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        rprint(f"  Synthetic examples loaded: {len(synthetic_rows)}")
    else:
        rprint("[yellow]No synthetic data found — training on real data only.[/yellow]")
        rprint("  Run 'crasis generate' first, or pass --synthetic-data to include synthetic examples.")

    # ------------------------------------------------------------------
    # Merge with oversampling
    # ------------------------------------------------------------------
    oversampled_real = real_rows * real_weight
    merged = synthetic_rows + oversampled_real
    random.shuffle(merged)

    total = len(merged)
    rprint(f"\n  Merged dataset : {total} examples")
    rprint(f"    Synthetic    : {len(synthetic_rows)}")
    rprint(f"    Real (x{real_weight})   : {len(oversampled_real)}")

    # Write merged data to a temp file for training
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as tmp:
        for row in merged:
            tmp.write(json.dumps(row) + "\n")
        merged_path = P(tmp.name)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    # Timestamped output dir so original model is never overwritten
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_output = P(output) / f"{crasis_spec.name}-mix-{timestamp}"

    rprint(f"\n[bold]Training...[/bold]")
    try:
        train_result = _train(
            spec=crasis_spec,
            data_path=merged_path,
            output_dir=run_output,
            device=device,
        )
    finally:
        merged_path.unlink(missing_ok=True)

    _print_train_result(train_result)

    if not train_result.passed_quality_gate:
        rprint("[bold red]✗ Quality gate failed. Aborting export.[/bold red]")
        rprint("  Try: collecting more real examples, adjusting --real-weight, or re-running generate.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    rprint("\n[bold]Exporting to ONNX...[/bold]")
    export_result = _export(
        spec=crasis_spec,
        model_path=train_result.model_path,
        output_dir=run_output,
    )

    size_color = "green" if export_result.within_size_constraint else "yellow"
    rprint(f"\n[bold green]✓ Mix complete![/bold green]")
    rprint(f"  Model : {export_result.onnx_path}")
    rprint(f"  Size  : [{size_color}]{export_result.model_size_mb:.1f}MB[/{size_color}]")
    rprint(f"  Run   : crasis classify --model {export_result.package_dir} \"your text here\"")


# ---------------------------------------------------------------------------
# crasis pull
# ---------------------------------------------------------------------------

REGISTRY_API = "https://api.github.com/repos/crasis-ai/crasis/releases/latest"


def _pull_specialist_sync(
    name: str,
    cache_dir: Path,
    force: bool = False,
    progress_callback: "Callable[[str], None] | None" = None,
) -> Path:
    """
    Download and extract a specialist package from the Crasis registry.

    Args:
        name: Specialist name (e.g. "sentiment-gate").
        cache_dir: Root cache directory; specialist is placed in cache_dir/name/.
        force: Re-download even if already cached.
        progress_callback: Optional callable called with status strings during download.

    Returns:
        Path to the extracted specialist package directory.

    Raises:
        SystemExit: On network errors or missing assets (when called from CLI context).
        RuntimeError: On network errors or missing assets (when called from non-CLI context).
    """
    import tarfile

    import requests

    def _emit(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    dest_dir = cache_dir / name
    onnx_file = dest_dir / f"{name}.onnx"

    if onnx_file.exists() and not force:
        _emit(f"Already cached: {dest_dir}")
        return dest_dir  # hot-reload path: package is on disk, return immediately

    dest_dir.mkdir(parents=True, exist_ok=True)

    _emit(f"Fetching release metadata for '{name}'...")
    try:
        resp = requests.get(REGISTRY_API, timeout=10)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Network error: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"GitHub API returned {resp.status_code}. Check your network or try again later."
        )

    assets: dict[str, str] = {a["name"]: a["browser_download_url"] for a in resp.json().get("assets", [])}

    expected = {
        f"{name}.onnx": dest_dir / f"{name}.onnx",
        f"{name}-label_map.json": dest_dir / "label_map.json",
        f"{name}-crasis_meta.json": dest_dir / "crasis_meta.json",
        f"{name}-tokenizer.tar.gz": dest_dir / f"{name}-tokenizer.tar.gz",
    }

    missing = [k for k in expected if k not in assets]
    if missing:
        raise RuntimeError(
            f"Specialist '{name}' not found in the latest release. "
            f"Missing assets: {', '.join(missing)}"
        )

    for asset_name, dest_path in expected.items():
        url = assets[asset_name]
        _emit(f"Downloading {asset_name}...")
        try:
            with requests.get(url, stream=True, timeout=60) as dl:
                dl.raise_for_status()
                with open(dest_path, "wb") as fh:
                    for chunk in dl.iter_content(chunk_size=8192):
                        fh.write(chunk)
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Download failed: {exc}") from exc

    # Extract tokenizer tarball
    tarball = dest_dir / f"{name}-tokenizer.tar.gz"
    tokenizer_dir = dest_dir / "tokenizer"
    _emit("Extracting tokenizer...")
    try:
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(tokenizer_dir)
    except tarfile.TarError as exc:
        import shutil
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to extract tokenizer: {exc}") from exc
    tarball.unlink()

    _emit(f"Specialist '{name}' ready at {dest_dir}")
    return dest_dir


@cli.command()
@click.argument("name")
@click.option("--force", "-f", is_flag=True, default=False, help="Re-download even if already cached")
@click.option("--cache-dir", default=None, help="Override default cache (~/.crasis/specialists/)")
def pull(name: str, force: bool, cache_dir: str | None) -> None:
    """Download a pre-built specialist from the Crasis registry."""
    from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TextColumn

    resolved_cache = Path(cache_dir) if cache_dir else Path.home() / ".crasis" / "specialists"
    dest_dir = resolved_cache / name
    onnx_file = dest_dir / f"{name}.onnx"

    if onnx_file.exists() and not force:
        rprint(f"[bold green]Already cached:[/bold green] {dest_dir}")
        rprint("  Use [bold]--force[/bold] to re-download.")
        return

    rprint(f"\n[bold cyan]Crasis — Pulling '{name}'[/bold cyan]")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), DownloadColumn()) as progress:
        task = progress.add_task("  Downloading...", total=None)

        def _on_progress(msg: str) -> None:
            progress.update(task, description=f"  {msg}")

        try:
            _pull_specialist_sync(
                name=name,
                cache_dir=resolved_cache,
                force=force,
                progress_callback=_on_progress,
            )
        except RuntimeError as exc:
            rprint(f"\n[bold red]Error:[/bold red] {exc}")
            sys.exit(1)

    rprint(f"\n[bold green]Specialist '{name}' ready.[/bold green]")
    rprint(f"  crasis classify --model {dest_dir} \"your text here\"")


# ---------------------------------------------------------------------------
# crasis mcp
# ---------------------------------------------------------------------------


@cli.command("mcp")
@click.option("--models-dir", default=None, help="Specialist cache directory. Default: ~/.crasis/specialists/")
@click.option("--api-key", envvar="OPENROUTER_API_KEY", default=None, help="OpenRouter API key (for build_specialist tool)")
@click.option("--data-dir", default=None, help="Training data directory. Default: ~/.crasis/data/")
def mcp_serve(models_dir: str | None, api_key: str | None, data_dir: str | None) -> None:
    """Start the Crasis MCP server over stdio."""
    import asyncio
    from crasis.mcp_server import run_server

    resolved_models = Path(models_dir) if models_dir else Path.home() / ".crasis" / "specialists"
    resolved_data = Path(data_dir) if data_dir else Path.home() / ".crasis" / "data"
    resolved_models.mkdir(parents=True, exist_ok=True)
    resolved_data.mkdir(parents=True, exist_ok=True)

    asyncio.run(run_server(models_dir=resolved_models, api_key=api_key, data_dir=resolved_data))


# ---------------------------------------------------------------------------
# crasis mcp-config
# ---------------------------------------------------------------------------


@cli.command("mcp-config")
@click.option("--models-dir", default=None, help="Specialist cache directory. Default: ~/.crasis/specialists/")
@click.option("--executable", default=None, help="Path to crasis executable. Default: auto-detected from PATH")
@click.option("--api-key", envvar="OPENROUTER_API_KEY", default=None, help="Bake OpenRouter API key into config env block")
@click.option("--output", "-o", default=None, help="Write JSON to file instead of stdout")
def mcp_config(models_dir: str | None, executable: str | None, api_key: str | None, output: str | None) -> None:
    """
    Print the Claude Desktop MCP server config block for Crasis.

    Pipe into your claude_desktop_config.json, or use --output to write directly.

    Example:

        crasis mcp-config | jq '.crasis'

        crasis mcp-config --output /tmp/crasis-mcp.json
    """
    import json
    import shutil
    from crasis.mcp_server import generate_claude_desktop_config

    resolved_models = Path(models_dir) if models_dir else Path.home() / ".crasis" / "specialists"

    if executable:
        resolved_executable = executable
    else:
        resolved_executable = shutil.which("crasis") or "crasis"

    cfg = generate_claude_desktop_config(
        models_dir=resolved_models,
        crasis_executable=resolved_executable,
        api_key=api_key,
    )

    formatted = json.dumps(cfg, indent=2)

    if output:
        Path(output).write_text(formatted + "\n", encoding="utf-8")
        rprint(f"[bold green]Config written to {output}[/bold green]")
    else:
        print(formatted)


if __name__ == "__main__":
    cli()
