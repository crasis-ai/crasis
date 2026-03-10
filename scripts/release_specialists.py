"""
scripts/release_specialists.py — Package and upload pre-built specialists to a GitHub Release.

For each specialist package directory, this script:
  1. Validates the expected files are present
  2. Creates {name}-tokenizer.tar.gz from the tokenizer/ subdirectory
  3. Renames label_map.json / crasis_meta.json to the {name}- prefixed form
  4. Creates (or updates) a GitHub Release via `gh`
  5. Uploads all assets, skipping any that already exist (unless --force)

Usage:
    python scripts/release_specialists.py --tag v1.0.0 models/email-urgency-onnx models/spam-filter-onnx
    python scripts/release_specialists.py --tag v1.0.0 --all-models
    python scripts/release_specialists.py --tag v1.0.0 --all-models --force
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

logging.basicConfig(format="%(levelname)-8s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
MODELS_DIR = REPO_ROOT / "models"

REQUIRED_FILES = ["{name}.onnx", "label_map.json", "crasis_meta.json", "tokenizer"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_package(pkg_dir: Path, name: str) -> None:
    """Raise if any required file is missing from the package directory."""
    missing = []
    for pattern in REQUIRED_FILES:
        path = pkg_dir / pattern.format(name=name)
        if not path.exists():
            missing.append(pattern.format(name=name))
    if missing:
        raise FileNotFoundError(
            f"Package '{pkg_dir}' is missing required files: {', '.join(missing)}"
        )


# ---------------------------------------------------------------------------
# Asset staging
# ---------------------------------------------------------------------------


def _stage_assets(pkg_dir: Path, name: str, staging_dir: Path) -> list[Path]:
    """
    Copy and rename package files into staging_dir using the release asset naming convention.

    Returns list of staged file paths ready for upload.
    """
    assets: list[Path] = []

    # {name}.onnx — no rename needed
    onnx_src = pkg_dir / f"{name}.onnx"
    onnx_dst = staging_dir / f"{name}.onnx"
    shutil.copy2(onnx_src, onnx_dst)
    assets.append(onnx_dst)
    logger.info("  Staged %s (%.2f MB)", onnx_dst.name, onnx_dst.stat().st_size / 1024 / 1024)

    # label_map.json → {name}-label_map.json
    lm_dst = staging_dir / f"{name}-label_map.json"
    shutil.copy2(pkg_dir / "label_map.json", lm_dst)
    assets.append(lm_dst)
    logger.info("  Staged %s", lm_dst.name)

    # crasis_meta.json → {name}-crasis_meta.json
    meta_dst = staging_dir / f"{name}-crasis_meta.json"
    shutil.copy2(pkg_dir / "crasis_meta.json", meta_dst)
    assets.append(meta_dst)
    logger.info("  Staged %s", meta_dst.name)

    # tokenizer/ → {name}-tokenizer.tar.gz
    tarball_dst = staging_dir / f"{name}-tokenizer.tar.gz"
    tok_dir = pkg_dir / "tokenizer"
    with tarfile.open(tarball_dst, "w:gz") as tf:
        for f in sorted(tok_dir.iterdir()):
            tf.add(f, arcname=f.name)
    assets.append(tarball_dst)
    logger.info(
        "  Staged %s (%.1f KB)",
        tarball_dst.name,
        tarball_dst.stat().st_size / 1024,
    )

    return assets


# ---------------------------------------------------------------------------
# GitHub release management
# ---------------------------------------------------------------------------


def _gh(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a gh command, raise on non-zero exit."""
    cmd = ["gh", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def _release_exists(tag: str) -> bool:
    result = _gh("release", "view", tag)
    return result.returncode == 0


def _existing_assets(tag: str) -> set[str]:
    """Return the set of asset filenames already attached to the release."""
    result = _gh("release", "view", tag, "--json", "assets")
    if result.returncode != 0:
        return set()
    data = json.loads(result.stdout)
    return {a["name"] for a in data.get("assets", [])}


def _create_release(tag: str, notes: str) -> None:
    logger.info("Creating release %s (draft)...", tag)
    result = _gh(
        "release", "create", tag,
        "--title", tag,
        "--notes", notes,
        "--draft",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create release: {result.stderr.strip()}")


def _publish_release(tag: str) -> None:
    logger.info("Publishing release %s...", tag)
    result = _gh("release", "edit", tag, "--draft=false")
    if result.returncode != 0:
        raise RuntimeError(f"Failed to publish release: {result.stderr.strip()}")


def _upload_asset(tag: str, path: Path) -> None:
    result = _gh("release", "upload", tag, str(path))
    if result.returncode != 0:
        raise RuntimeError(f"Failed to upload {path.name}: {result.stderr.strip()}")


# ---------------------------------------------------------------------------
# Per-specialist release flow
# ---------------------------------------------------------------------------


def _release_specialist(pkg_dir: Path, name: str, tag: str, force: bool) -> None:
    logger.info("Processing '%s' from %s", name, pkg_dir)

    _validate_package(pkg_dir, name)

    existing = _existing_assets(tag) if _release_exists(tag) else set()

    with tempfile.TemporaryDirectory(prefix=f"crasis-release-{name}-") as tmp:
        staging = Path(tmp)
        assets = _stage_assets(pkg_dir, name, staging)

        for asset in assets:
            if asset.name in existing and not force:
                logger.info("  Skipping %s (already uploaded, use --force to overwrite)", asset.name)
                continue
            if asset.name in existing and force:
                logger.info("  Deleting existing %s before re-upload...", asset.name)
                _gh("release", "delete-asset", tag, asset.name, "--yes")
            logger.info("  Uploading %s...", asset.name)
            _upload_asset(tag, asset)

    logger.info("  Done: '%s'", name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _infer_name(pkg_dir: Path) -> str:
    """Infer specialist name from package directory (strips -onnx suffix)."""
    return pkg_dir.name.removesuffix("-onnx")


def _discover_all_packages() -> list[Path]:
    """Find all *-onnx directories under models/."""
    return sorted(MODELS_DIR.glob("*-onnx"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload specialist ONNX packages to a GitHub Release.")
    parser.add_argument("--tag", required=True, help="Release tag, e.g. v1.0.0")
    parser.add_argument(
        "--notes",
        default="Pre-built specialist models. Install with: crasis pull <name>",
        help="Release notes (used only when creating a new release)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help=f"Upload all *-onnx packages found in {MODELS_DIR}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upload assets that already exist in the release",
    )
    parser.add_argument(
        "packages",
        nargs="*",
        metavar="PKG_DIR",
        help="Explicit package directories to upload (e.g. models/email-urgency-onnx)",
    )
    args = parser.parse_args()

    if not args.packages and not args.all_models:
        parser.error("Provide at least one PKG_DIR or use --all-models")

    pkg_dirs: list[Path] = []
    if args.all_models:
        pkg_dirs = _discover_all_packages()
        if not pkg_dirs:
            logger.error("No *-onnx directories found in %s", MODELS_DIR)
            sys.exit(1)
        logger.info("Discovered %d packages: %s", len(pkg_dirs), [p.name for p in pkg_dirs])
    else:
        pkg_dirs = [Path(p) for p in args.packages]

    for pkg_dir in pkg_dirs:
        if not pkg_dir.exists():
            logger.error("Package directory not found: %s", pkg_dir)
            sys.exit(1)

    # Create release if it doesn't exist yet
    if not _release_exists(args.tag):
        _create_release(args.tag, args.notes)
    else:
        logger.info("Release %s already exists, uploading assets.", args.tag)

    errors: list[str] = []
    for pkg_dir in pkg_dirs:
        name = _infer_name(pkg_dir)
        try:
            _release_specialist(pkg_dir, name, args.tag, args.force)
        except Exception as exc:
            logger.error("Failed to release '%s': %s", name, exc)
            errors.append(name)

    if errors:
        logger.error("Failed specialists: %s", errors)
        sys.exit(1)

    _publish_release(args.tag)
    logger.info("All specialists uploaded to release %s.", args.tag)
    logger.info("Users can now run: crasis pull <name>")


if __name__ == "__main__":
    main()
