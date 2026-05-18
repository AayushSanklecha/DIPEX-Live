"""
scripts/deploy_to_hf.py
-----------------------
Deploys the DIPEX project to a Hugging Face Space (Docker SDK).

Strategy:
  1. Upload the main codebase (respecting .gitignore exclusions for secrets/venvs)
  2. Explicitly upload audit/ and reports/ directories which are in .gitignore
     but MUST be included so the Space has historical pipeline run history baked in.
"""

import os
from pathlib import Path
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

# Directories that are in .gitignore but must be deployed
FORCE_INCLUDE_DIRS = ["audit", "reports"]

# Files/patterns to always exclude
IGNORE_PATTERNS = [
    # Version control & IDE
    ".git/*",
    ".vscode/*",
    # Python environments (rebuilt inside Docker)
    "venv/*",
    "env/*",
    ".venv/*",
    "__pycache__/*",
    "*/__pycache__/*",
    "**/__pycache__/*",
    # Secrets — NEVER deploy
    ".env",
    # Frontend source (pre-built into dashboard/)
    "frontend/node_modules/*",
    "frontend/src/*",
    "frontend/public/*",
    # Large raw data files not needed at runtime
    "data/*",
    "Invistico_Airline.csv",
    "mock_banking_data.csv",
    # Scratch / temp
    "scratch/*",
    "logs.txt",
    # Office docs (not needed at runtime)
    "*.docx",
    "*.pdf",
    "*.pptx",
    # Local DB (ephemeral dev artifact)
    "Mock_DIPEX_Database.db",
    "sample_company.db",
]


def deploy_to_hf():
    token = os.environ.get("HF_API_KEY")
    if not token:
        print("Error: HF_API_KEY not found in .env")
        return

    api = HfApi(token=token)

    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"Authenticated as Hugging Face user: {username}")
    except Exception as e:
        print(f"Error authenticating with Hugging Face: {e}")
        print("Please ensure your HF_API_KEY is valid and has Write permissions.")
        return

    repo_id = f"{username}/DIPEX-Live"

    print(f"Creating Hugging Face Space: {repo_id}...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
        print("Space created or already exists.")
    except Exception as e:
        print(f"Error creating Space: {e}")
        return

    # ── Step 1: Upload main codebase ──────────────────────────────────────────
    print(f"\nStep 1/2 — Uploading main codebase to {repo_id}...")
    try:
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=IGNORE_PATTERNS,
        )
        print("Main codebase uploaded.")
    except Exception as e:
        print(f"Error uploading codebase: {e}")
        return

    # ── Step 2: Force-upload audit/ and reports/ (bypasses .gitignore) ────────
    print(f"\nStep 2/2 — Force-uploading historical data (audit/ + reports/)...")
    root = Path(".")
    for dir_name in FORCE_INCLUDE_DIRS:
        dir_path = root / dir_name
        if not dir_path.is_dir():
            print(f"  Skipping {dir_name}/ — directory not found")
            continue

        files = list(dir_path.rglob("*"))
        files = [f for f in files if f.is_file()]
        print(f"  Uploading {dir_name}/ ({len(files)} files)...")

        for fpath in files:
            rel = fpath.relative_to(root)
            try:
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=str(rel).replace("\\", "/"),
                    repo_id=repo_id,
                    repo_type="space",
                )
            except Exception as e:
                print(f"    Warning: failed to upload {rel}: {e}")

        print(f"  {dir_name}/ uploaded.")

    print(f"\nDeployed successfully to https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    deploy_to_hf()
