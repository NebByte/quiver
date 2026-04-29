import os
import sys
import shutil
import subprocess
import json
import time

def run_command(cmd, capture=False):
    if capture:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    subprocess.run(cmd, shell=True, check=True)

def main():
    print("==================================================")
    print("    QUIVER LIVE DEMO CLI (ARM CLOUD EXECUTION)    ")
    print("==================================================")
    print("This tool will upload your dataset to an AWS Graviton")
    print("ARM server, build Quiver, and compare it to SQLite.")
    print("")

    file_path = input("Drag and drop a CSV file here (or type the path): ").strip().strip("'\"")

    if not os.path.exists(file_path):
        print(f"Error: Could not find file {file_path}")
        sys.exit(1)

    print("\n[1/5] Preparing dataset...")
    # Copy file to demo/test_db.csv
    os.makedirs("demo", exist_ok=True)
    shutil.copy(file_path, "demo/test_db.csv")

    print("[2/5] Uploading to ARM Neoverse Cloud...")
    try:
        run_command('git add demo/test_db.csv')
        run_command('git commit -m "Upload user DB for demo benchmark"')
        run_command('git pull --rebase origin main', capture=True)
        run_command('git push origin main')
    except subprocess.CalledProcessError:
        print("Git push failed. Ensure you have no unstaged changes and are online.")
        sys.exit(1)

    print("[3/5] Starting execution on 128-bit NEON hardware...")
    time.sleep(10) # Give GitHub a moment to register the run

    # Get the latest run ID for the demo workflow
    try:
        run_id = run_command("gh run list --workflow=demo-arm.yml --limit 1 --json databaseId -q '.[0].databaseId'", capture=True)
        if not run_id:
            print("Could not find the GitHub Action run.")
            sys.exit(1)
    except Exception as e:
        print("Error checking GitHub Actions (is the 'gh' CLI installed and authenticated?).", e)
        sys.exit(1)

    print(f"[4/5] Waiting for benchmark to complete (Run ID: {run_id}). This takes ~1 minute...")
    
    while True:
        status = run_command(f"gh run view {run_id} --json status -q .status", capture=True)
        if status == "completed":
            break
        print("      ... still running ...")
        time.sleep(10)

    print("[5/5] Downloading empirical results...")
    os.makedirs("proofs", exist_ok=True)
    
    # Download the artifact
    try:
        run_command(f"gh run download {run_id} -n demo-results -D proofs/")
    except Exception:
        print("Warning: Could not download artifact via gh CLI. It might have failed.")
    
    result_path = "proofs/demo_results.json"
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            data = json.load(f)
            quiver_ms = data["quiver_ns"] / 1_000_000
            sqlite_ms = data["sqlite_ns"] / 1_000_000
            speedup = data["speedup"]

            print("\n==================================================")
            print("                 OFFICIAL RESULTS                 ")
            print("==================================================")
            print(f"Dataset indexed on physical ARM Neoverse hardware.")
            print(f"SQLite Query Time: {sqlite_ms:.4f} ms")
            print(f"Quiver Query Time: {quiver_ms:.4f} ms")
            print("--------------------------------------------------")
            print(f"🔥 QUIVER IS {speedup:.1f}X FASTER THAN SQLITE 🔥")
            print("==================================================")
            print("The raw proof has been saved to the proofs/ folder.")
    else:
        print("Error: Results file was not generated. The action likely failed.")

if __name__ == "__main__":
    main()
