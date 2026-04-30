#!/usr/bin/env python3
"""Quiver demo CLI.

Drag-and-drop a SQLite .db file. The CLI uploads it to a transient GitHub
release, dispatches the `demo-arm.yml` workflow on a real aarch64 runner,
waits for results, saves them to `business/proofs/`, and prints how many
times faster Quiver is than indexed SQLite for a 2-column AND filter.

Requires:
  - `gh` CLI on PATH and authenticated (`gh auth login`)
  - the working directory inside the quiver repo (or run from anywhere;
    the script resolves paths relative to itself)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from shutil import which

REPO_ROOT = Path(__file__).resolve().parent.parent
PROOFS_DIR = REPO_ROOT / "business" / "proofs"
WORKFLOW_FILE = "demo-arm.yml"
ARTIFACT_NAME = "demo-results"


def banner() -> None:
    print()
    print("==============================================================")
    print("              QUIVER DEMO  -  Real ARM Hardware              ")
    print("       Drop a SQLite DB. We will benchmark it on aarch64.    ")
    print("==============================================================")
    print()


def run(cmd, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def must(cmd) -> str:
    cp = run(cmd)
    if cp.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{cp.stderr.strip()}")
    return cp.stdout.strip()


def need(cmd: str) -> None:
    if which(cmd) is None:
        sys.exit(f"error: '{cmd}' is required on PATH. Install it and retry.")


QUOTE_PAIRS = [('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")]


def clean_path_input(s: str) -> str:
    s = s.strip()
    # PowerShell drag-drop sometimes pastes "& 'C:\path\file'" (call-operator form).
    if s.startswith("& "):
        s = s[2:].strip()
    # Strip matching quote pairs, including curly/smart quotes.
    for left, right in QUOTE_PAIRS:
        if s.startswith(left) and s.endswith(right) and len(s) >= 2:
            s = s[1:-1]
            break
    # Expand ~ and %ENV% / $ENV.
    s = os.path.expandvars(os.path.expanduser(s))
    # PowerShell escapes spaces with backticks; remove the escape, keep the space.
    s = s.replace("` ", " ")
    return s.strip()


def prompt_db_file() -> Path:
    while True:
        try:
            raw = input("Drag and drop your .db file here, then press Enter:\n> ")
        except (EOFError, KeyboardInterrupt):
            sys.exit("\naborted.")
        if not raw.strip():
            print("  ! empty input\n")
            continue
        cleaned = clean_path_input(raw)
        path = Path(cleaned)
        if not path.exists():
            print(f"  ! file not found")
            print(f"      raw input  : {raw!r}")
            print(f"      cleaned    : {cleaned!r}")
            print(f"      resolved   : {path.resolve(strict=False)}")
            print(
                "    tip: in PowerShell, type the path inside single quotes, "
                "or drag from File Explorer (not from inside a zip/OneDrive "
                "placeholder).\n"
            )
            continue
        if not path.is_file():
            print(f"  ! not a file: {path}\n")
            continue
        return path


def get_repo() -> str:
    cp = run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        cwd=str(REPO_ROOT),
    )
    if cp.returncode != 0:
        sys.exit(
            "could not resolve GitHub repo from "
            f"{REPO_ROOT} (gh said: {cp.stderr.strip()})"
        )
    return cp.stdout.strip()


def create_release_with_db(repo: str, db_file: Path, tag: str) -> None:
    print(f"  > uploading DB as release '{tag}' to {repo} ...")
    with tempfile.TemporaryDirectory() as td:
        upload_path = Path(td) / "input.db"
        shutil.copy2(db_file, upload_path)
        cp = run([
            "gh", "release", "create", tag,
            str(upload_path),
            "--repo", repo,
            "--title", f"Quiver demo {tag}",
            "--notes", "Transient demo input. Auto-deleted after the run completes.",
        ])
        if cp.returncode != 0:
            sys.exit(f"failed to create release:\n{cp.stderr}")


def trigger_workflow(repo: str, tag: str, run_id: str) -> None:
    print(f"  > dispatching {WORKFLOW_FILE} on real ARM hardware ...")
    cp = run([
        "gh", "workflow", "run", WORKFLOW_FILE,
        "--repo", repo,
        "-f", f"release_tag={tag}",
        "-f", f"run_id={run_id}",
    ])
    if cp.returncode != 0:
        sys.exit(f"failed to dispatch workflow:\n{cp.stderr}")


def find_run(repo: str, run_id: str, dispatched_after: float) -> int:
    deadline = time.time() + 90
    target_title = f"demo {run_id}"
    while time.time() < deadline:
        cp = run([
            "gh", "run", "list",
            "--repo", repo,
            "--workflow", WORKFLOW_FILE,
            "-L", "10",
            "--json", "databaseId,createdAt,displayTitle,event",
        ])
        if cp.returncode == 0:
            try:
                runs = json.loads(cp.stdout)
            except json.JSONDecodeError:
                runs = []
            for r in runs:
                if r.get("displayTitle") == target_title:
                    return int(r["databaseId"])
            for r in runs:
                created = datetime.fromisoformat(
                    r["createdAt"].replace("Z", "+00:00")
                ).timestamp()
                if r.get("event") == "workflow_dispatch" and created >= dispatched_after - 5:
                    return int(r["databaseId"])
        time.sleep(3)
    sys.exit("error: timed out waiting for the workflow run to appear")


def wait_for_run(repo: str, db_id: int) -> str:
    print(f"  > waiting for run {db_id} (typically 2-5 minutes) ...")
    last_status = ""
    while True:
        cp = run([
            "gh", "run", "view", str(db_id),
            "--repo", repo,
            "--json", "status,conclusion,url",
        ])
        if cp.returncode != 0:
            time.sleep(5)
            continue
        info = json.loads(cp.stdout)
        status = info["status"]
        if status != last_status:
            print(f"    status: {status}")
            last_status = status
        if status == "completed":
            print(f"    conclusion: {info['conclusion']}")
            print(f"    url: {info['url']}")
            return info["conclusion"]
        time.sleep(10)


def download_artifact(repo: str, db_id: int, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    cp = run([
        "gh", "run", "download", str(db_id),
        "--repo", repo,
        "-n", ARTIFACT_NAME,
        "-D", str(dest),
    ])
    if cp.returncode != 0:
        sys.exit(f"failed to download artifact:\n{cp.stderr}")
    candidate = dest / "demo_results.json"
    if not candidate.exists():
        sys.exit(f"artifact missing demo_results.json in {dest}")
    return candidate


def cleanup_release(repo: str, tag: str) -> None:
    run(["gh", "release", "delete", tag, "--repo", repo, "--yes", "--cleanup-tag"])


def parse_results_jsonl(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").strip()
    last_line = text.splitlines()[-1] if text else ""
    return json.loads(last_line)


def print_results(results: dict, db_file: Path, flat_path: Path) -> None:
    speedup = results.get("speedup_multiplier", 0.0)
    quiver_ns = int(results.get("quiver_time_ns", 0))
    sqlite_ns = int(results.get("sqlite_time_ns", 0))
    rows = int(results.get("rows_processed", 0))
    arch = results.get("arch", "?")

    print()
    print("==============================================================")
    print("                          RESULTS                            ")
    print("==============================================================")
    print(f"  DB file        : {results.get('db_file', db_file.name)}")
    print(f"  table          : {results.get('table', '?')}")
    print(f"  columns        : {results.get('col1', '?')} AND {results.get('col2', '?')}")
    print(f"  rows processed : {rows:,}")
    print(f"  hardware       : {arch} (GitHub Actions ubuntu-24.04-arm)")
    print(f"  Quiver         : {quiver_ns:>12,} ns / query")
    print(f"  SQLite         : {sqlite_ns:>12,} ns / query")
    print()
    print(f"  >>> Quiver is {speedup:.1f}x faster than SQLite <<<")
    print()
    print(f"  full result saved to: {flat_path}")
    print()


def main() -> int:
    banner()
    need("gh")

    db_file = prompt_db_file()
    print(f"  > using DB: {db_file}\n")

    repo = get_repo()
    print(f"  > GitHub repo: {repo}")

    tag = f"demo-{int(time.time())}-{os.getpid()}"
    run_id = tag

    create_release_with_db(repo, db_file, tag)
    dispatched_at = time.time()
    try:
        trigger_workflow(repo, tag, run_id)
        db_id = find_run(repo, run_id, dispatched_at)
        print(f"  > workflow run id: {db_id}")
        conclusion = wait_for_run(repo, db_id)
        if conclusion != "success":
            sys.exit(
                f"workflow did not succeed (conclusion={conclusion}). "
                f"check the Actions tab on {repo}."
            )

        out_dir = PROOFS_DIR / f"demo_{tag}"
        results_path = download_artifact(repo, db_id, out_dir)
        results = parse_results_jsonl(results_path)
    finally:
        cleanup_release(repo, tag)

    flat_path = PROOFS_DIR / f"demo_{tag}.json"
    flat_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_results(results, db_file, flat_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
