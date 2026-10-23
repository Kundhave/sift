"""Create/update the Hugging Face Space and upload the Sift deployment package.
Usage (after `hf auth login` or `huggingface-cli login`):  HF_USERNAME=<you> .venv/bin/python deploy/deploy_hf.py [--name sift] [--private]
Uploads: Dockerfile (from deploy/Dockerfile), README.md (from deploy/README_SPACE.md), and only the runtime files the image COPYs."""
import argparse, os, sys
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="sift")
ap.add_argument("--private", action="store_true")
a = ap.parse_args()
user = os.environ.get("HF_USERNAME")
if not user:
    sys.exit("set HF_USERNAME=<your Hugging Face username>")
api = HfApi()
try:
    who = api.whoami()["name"]
except Exception as e:
    sys.exit(f"not logged in ({type(e).__name__}). Run: hf auth login   (token with WRITE access)")
if who != user:
    print(f"note: logged in as {who!r}, HF_USERNAME={user!r}; using HF_USERNAME as the namespace")
repo = f"{user}/{a.name}"
api.create_repo(repo, repo_type="space", space_sdk="docker", private=a.private, exist_ok=True)
api.upload_file(path_or_fileobj=str(ROOT / "deploy/README_SPACE.md"), path_in_repo="README.md", repo_id=repo, repo_type="space")
api.upload_file(path_or_fileobj=str(ROOT / "deploy/Dockerfile"), path_in_repo="Dockerfile", repo_id=repo, repo_type="space")
api.upload_file(path_or_fileobj=str(ROOT / ".dockerignore"), path_in_repo=".dockerignore", repo_id=repo, repo_type="space")
api.upload_file(path_or_fileobj=str(ROOT / "deploy/requirements-deploy.txt"), path_in_repo="deploy/requirements-deploy.txt", repo_id=repo, repo_type="space")
need = ["src/**", "app/__init__.py", "app/backend/**", "app/frontend/**", "artifacts/full/*", "artifacts/student/*",
        "artifacts/student_onnx_int8/*", "artifacts/kmeans/*", "results/**", "experiments/results/*.json",
        "experiments/results/*.csv", "experiments/results/*.png", "experiments/results/*.md", "experiments/results/*.txt",
        "data/dataset_statistics.json", "data/sift_ia_v1_train.csv", "data/sift_ia_v1_val.csv", "data/sift_ia_v1_test.csv"]
api.upload_folder(folder_path=str(ROOT), repo_id=repo, repo_type="space", allow_patterns=need,
                  ignore_patterns=["**/node_modules/**", "**/dist/**", "**/__pycache__/**", "**/*.pyc"],
                  commit_message="Deploy Sift")
url = f"https://huggingface.co/spaces/{repo}"
(ROOT / "deploy/DEPLOY_URL.txt").write_text(url + "\n")
print(f"Uploaded. The Space is now building (first build takes ~10-20 min): {url}")
print(f"Live app URL once running: https://{user}-{a.name}.hf.space")
