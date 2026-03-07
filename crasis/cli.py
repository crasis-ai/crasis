"""
crasis.cli — Command-line interface.

Commands:
  crasis generate  — Generate synthetic training data
  crasis train     — Train a specialist model
  crasis eval      — Evaluate a trained model
  crasis export    — Export to ONNX
  crasis classify  — Run inference on text input(s)
  crasis build     — Full pipeline: generate → train → eval → export
"""

from __future__ import annotations

import logging
import sys

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

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


if __name__ == "__main__":
    cli()
