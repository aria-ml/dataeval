#!/usr/bin/env python3
"""
Check if jupyter-cache is up to date for all notebooks in the documentation.

This script checks each notebook against the jupyter-cache to determine if it needs re-execution.
It computes a hash of each notebook (excluding outputs) and checks if a matching cached version exists.

Usage:
    python check_notebook_cache.py          # Check cache status
    python check_notebook_cache.py --clean  # Remove stale cache entries
    python check_notebook_cache.py --fix    # Fix inconsistent cache state

    # Or with uv:
    uv run python check_notebook_cache.py
    uv run python check_notebook_cache.py --clean
    uv run python check_notebook_cache.py --fix

Exit codes:
    0 - All notebooks are cached and up to date (or cleanup succeeded)
    1 - One or more notebooks need execution (or cleanup found issues)
"""

import argparse
import hashlib
import shutil
import sqlite3
import sys
from pathlib import Path

import jupytext
import nbformat


def compute_notebook_hash(nb_path: Path) -> str:
    """
    Compute hash of notebook content using jupyter-cache's algorithm.

    This replicates the logic from jupyter_cache.JupyterCacheBase.create_hashed_notebook
    to match how myst-nb determines if a cached execution is still valid.
    """
    import copy

    nb = jupytext.read(nb_path)

    # Copy and normalize the notebook (same as jupyter-cache)
    nb = copy.deepcopy(nb)
    nb = nbformat.convert(nb, to_version=4)

    # Only include code cells (jupyter-cache removes markdown cells)
    nb.cells = [cell for cell in nb.cells if cell.cell_type == "code"]

    # Create hash notebook with only kernelspec metadata (default for jupyter-cache)
    nb_metadata = ("kernelspec",)
    cell_metadata = None

    hash_nb = nbformat.from_dict(
        {
            "nbformat": nb.nbformat,
            "nbformat_minor": 4,  # v4.5 includes cell ids, which are not cached
            "metadata": {k: v for k, v in nb.metadata.items() if nb_metadata is None or (k in nb_metadata)},
            "cells": [
                {
                    "cell_type": cell.cell_type,
                    "source": cell.source,
                    "metadata": {
                        k: v for k, v in cell.metadata.items() if cell_metadata is None or (k in cell_metadata)
                    },
                    "execution_count": None,
                    "outputs": [],
                }
                for cell in nb.cells
                if cell.cell_type == "code"
            ],
        },
    )

    # Hash using NO_CONVERT mode (same as jupyter-cache)
    nb_str = nbformat.writes(hash_nb, nbformat.NO_CONVERT)
    return hashlib.md5(nb_str.encode()).hexdigest()


def check_cache_status(docs_source_dir: Path = Path("docs/source")):
    """
    Check which notebooks are cached and which need execution.

    Args:
        docs_source_dir: Path to the docs/source directory containing notebooks/

    Returns
    -------
        True if all notebooks are cached, False otherwise
    """
    notebooks_dir = docs_source_dir / "notebooks"
    cache_dir = docs_source_dir / ".jupyter_cache" / "executed"

    if not notebooks_dir.exists():
        print(f"Error: {notebooks_dir} does not exist", file=sys.stderr)
        return False

    notebooks = sorted(notebooks_dir.glob("*.md"))

    if not notebooks:
        print(f"Warning: No notebooks found in {notebooks_dir}", file=sys.stderr)
        return True

    print(f"Found {len(notebooks)} notebooks")

    # Check if cache directory exists
    if not cache_dir.exists():
        print(f"\nWarning: Cache directory does not exist: {cache_dir}")
        print("All notebooks will need to be executed.")
        return False

    # Get all cached hashes
    cached_hashes = {p.name for p in cache_dir.iterdir() if p.is_dir()}
    print(f"Found {len(cached_hashes)} cached notebook executions\n")

    cached = []
    outdated = []
    not_cached = []

    print("Cache Status Check:")
    print("=" * 80)

    for nb_path in notebooks:
        nb_hash = compute_notebook_hash(nb_path)
        cache_path = cache_dir / nb_hash / "base.ipynb"

        is_cached = cache_path.exists()
        has_old_cache = any(nb_hash != h and (cache_dir / h / "base.ipynb").exists() for h in cached_hashes)

        if is_cached:
            status = "✓ CACHED"
            cached.append(nb_path.name)
        elif has_old_cache:
            status = "⚠ OUTDATED"
            outdated.append(nb_path.name)
        else:
            status = "✗ NOT CACHED"
            not_cached.append(nb_path.name)

        print(f"{status:15} {nb_path.name}")

    print("=" * 80)
    print("\nSummary:")
    print(f"  Up to date: {len(cached)}/{len(notebooks)}")
    print(f"  Outdated:   {len(outdated)}/{len(notebooks)}")
    print(f"  Not cached: {len(not_cached)}/{len(notebooks)}")

    needs_execution = outdated + not_cached

    if needs_execution:
        print(f"\nNotebooks that need execution ({len(needs_execution)}):")
        for nb in needs_execution:
            print(f"  - {nb}")
        print("\nTo update the cache, run: nox -e docs")
        print("To clear the cache completely, run: nox -e docs -- clean")
        return False
    print("\n✓ All notebooks are cached and up to date!")
    return True


def clean_stale_cache(docs_source_dir: Path = Path("docs/source"), dry_run: bool = False) -> int:
    """
    Remove stale cache entries that don't correspond to any current notebook.

    Args:
        docs_source_dir: Path to the docs/source directory containing notebooks/
        dry_run: If True, only report what would be deleted without actually deleting

    Returns
    -------
        Number of stale cache entries removed (or would be removed if dry_run)
    """
    notebooks_dir = docs_source_dir / "notebooks"
    cache_dir = docs_source_dir / ".jupyter_cache" / "executed"

    if not notebooks_dir.exists():
        print(f"Error: {notebooks_dir} does not exist", file=sys.stderr)
        return 0

    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        return 0

    notebooks = sorted(notebooks_dir.glob("*.md"))

    if not notebooks:
        print(f"Warning: No notebooks found in {notebooks_dir}", file=sys.stderr)
        return 0

    # Compute hashes for all current notebooks
    current_hashes = set()
    for nb_path in notebooks:
        nb_hash = compute_notebook_hash(nb_path)
        current_hashes.add(nb_hash)

    # Find all cache directories
    cache_dirs = [p for p in cache_dir.iterdir() if p.is_dir()]

    # Find stale cache entries (directories that don't match any current notebook hash)
    stale_dirs = [d for d in cache_dirs if d.name not in current_hashes]

    if not stale_dirs:
        print("No stale cache entries found.")
        return 0

    print(f"Found {len(stale_dirs)} stale cache entries:")
    for stale in stale_dirs:
        print(f"  - {stale.name}")

    if dry_run:
        print(f"\nDry run: would delete {len(stale_dirs)} stale cache entries")
    else:
        print(f"\nDeleting {len(stale_dirs)} stale cache entries...")
        for stale in stale_dirs:
            shutil.rmtree(stale)
            print(f"  Deleted: {stale.name}")
        print(f"✓ Removed {len(stale_dirs)} stale cache entries")

    return len(stale_dirs)


def _remove_orphan_db_records(
    cursor: sqlite3.Cursor,
    executed_dir: Path,
    db_hashes: dict[str, tuple[int, str]],
    dry_run: bool,
) -> int:
    """Remove database records that don't have corresponding folders on disk."""
    orphan_records = [
        (pk, hashkey, uri) for hashkey, (pk, uri) in db_hashes.items() if not (executed_dir / hashkey).exists()
    ]

    if not orphan_records:
        return 0

    print(f"Found {len(orphan_records)} database records without folders on disk:")
    for _pk, hashkey, uri in orphan_records:
        nb_name = Path(uri).name if uri else "unknown"
        print(f"  - {hashkey[:12]}... ({nb_name})")

    if dry_run:
        print(f"\nDry run: would remove {len(orphan_records)} orphan database records")
    else:
        print(f"\nRemoving {len(orphan_records)} orphan database records...")
        for pk, hashkey, _uri in orphan_records:
            cursor.execute("DELETE FROM nbcache WHERE pk = ?", (pk,))
            print(f"  Removed: {hashkey[:12]}...")
        print(f"Removed {len(orphan_records)} orphan records from database")

    return len(orphan_records)


def _remove_orphan_folders(executed_dir: Path, db_hashes: dict[str, tuple[int, str]], dry_run: bool) -> int:
    """Remove folders on disk that don't have corresponding database records."""
    disk_hashes = {p.name for p in executed_dir.iterdir() if p.is_dir()}
    orphan_folders = [hashkey for hashkey in disk_hashes if hashkey not in db_hashes]

    if not orphan_folders:
        return 0

    print(f"\nFound {len(orphan_folders)} folders on disk without database records:")
    for hashkey in orphan_folders:
        print(f"  - {hashkey}")

    if dry_run:
        print(f"\nDry run: would delete {len(orphan_folders)} orphan folders")
    else:
        print(f"\nDeleting {len(orphan_folders)} orphan folders...")
        for hashkey in orphan_folders:
            shutil.rmtree(executed_dir / hashkey)
            print(f"  Deleted: {hashkey}")
        print(f"Deleted {len(orphan_folders)} orphan folders")

    return len(orphan_folders)


def fix_cache_state(docs_source_dir: Path = Path("docs/source"), dry_run: bool = False) -> int:
    """
    Fix inconsistent cache state by reconciling the database with the filesystem.

    This handles two cases:
    1. Cache database has records for hashes whose folders don't exist on disk
       -> Remove these orphan records from the database
    2. Folders exist on disk but have no corresponding database record
       -> Remove these orphan folders from disk

    Args:
        docs_source_dir: Path to the docs/source directory
        dry_run: If True, only report what would be fixed without actually fixing

    Returns
    -------
        Number of issues fixed (or would be fixed if dry_run)
    """
    cache_dir = docs_source_dir / ".jupyter_cache"
    db_path = cache_dir / "global.db"
    executed_dir = cache_dir / "executed"

    if not db_path.exists():
        print(f"Cache database does not exist: {db_path}")
        return 0

    if not executed_dir.exists():
        print(f"Executed cache directory does not exist: {executed_dir}")
        return 0

    # Get all hashkeys from the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT pk, hashkey, uri FROM nbcache")
    db_records = cursor.fetchall()
    db_hashes = {row[1]: (row[0], row[2]) for row in db_records}  # hashkey -> (pk, uri)

    # Fix orphan records and folders
    issues_fixed = _remove_orphan_db_records(cursor, executed_dir, db_hashes, dry_run)
    if not dry_run and issues_fixed > 0:
        conn.commit()
    issues_fixed += _remove_orphan_folders(executed_dir, db_hashes, dry_run)

    conn.close()

    if issues_fixed == 0:
        print("Cache state is consistent. No fixes needed.")

    return issues_fixed


def find_docs_source_dir() -> Path | None:
    """
    Find the docs/source directory by searching up from the script location.

    Returns
    -------
        Path to docs/source directory, or None if not found
    """
    script_dir = Path(__file__).resolve().parent

    # Case 1: Script is in docs/ directory
    if script_dir.name == "docs":
        docs_source = script_dir / "source"
        if docs_source.exists():
            return docs_source

    # Case 2: Script is in docs/source/ directory
    if script_dir.name == "source" and script_dir.parent.name == "docs":
        return script_dir

    # Case 3: Script is in project root
    docs_source = script_dir / "docs" / "source"
    if docs_source.exists():
        return docs_source

    # Case 4: Search up the directory tree
    current = script_dir
    for _ in range(5):  # Search up to 5 levels
        docs_source = current / "docs" / "source"
        if docs_source.exists():
            return docs_source
        if current.parent == current:  # Reached filesystem root
            break
        current = current.parent

    return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check and manage jupyter-cache for documentation notebooks.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove stale cache entries that don't match any current notebook",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Fix inconsistent cache state (db records without folders, or folders without db records)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without actually changing (use with --clean or --fix)",
    )
    args = parser.parse_args()

    docs_source_dir = find_docs_source_dir()

    if docs_source_dir is None:
        print("Error: Could not find docs/source directory", file=sys.stderr)
        print("Please run this script from:", file=sys.stderr)
        print("  - Project root directory", file=sys.stderr)
        print("  - docs/ directory", file=sys.stderr)
        print("  - docs/source/ directory", file=sys.stderr)
        return 1

    print(f"Using docs/source at: {docs_source_dir}\n")

    if args.fix:
        fixed = fix_cache_state(docs_source_dir, dry_run=args.dry_run)
        return 0 if fixed >= 0 else 1
    if args.clean:
        removed = clean_stale_cache(docs_source_dir, dry_run=args.dry_run)
        return 0 if removed >= 0 else 1
    all_cached = check_cache_status(docs_source_dir)
    return 0 if all_cached else 1


if __name__ == "__main__":
    sys.exit(main())
