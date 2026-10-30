from huggingface_hub import create_repo, upload_file, HfApi
import os

repo_id = "grahmseinstein/tinyllm-28m"
api = HfApi()

print(f"Creating {repo_id}")
create_repo(repo_id, repo_type="model", exist_ok=True)

# Ensure checkpoint exists
ckpt = None
for p in ["checkpoints/tinyllm.pt", "tinyllm.pt"]:
    if os.path.exists(p):
        ckpt = p
        break

if ckpt is None:
    print("ERROR: No tinyllm.pt found! Run training first.")
    exit(1)

print(f"Found checkpoint: {ckpt}")

# README
readme = """---
library_name: pytorch
pipeline_tag: text-generation
tags:
- tinyllm
- gpt2
base_model: gpt2
---

# tinyllm-28m

28.92M params, GPT2 tokenizer, trained on Tiny Shakespeare.

By grahmseinstein
"""

open("README_HF.md","w").write(readme)

print("Uploading tinyllm.pt...")
upload_file(path_or_fileobj=ckpt, path_in_repo="tinyllm.pt", repo_id=repo_id, repo_type="model")

print("Uploading model.py...")
if os.path.exists("src/model.py"):
    upload_file(path_or_fileobj="src/model.py", path_in_repo="model.py", repo_id=repo_id, repo_type="model")

print("Uploading app.py...")
if os.path.exists("app.py"):
    upload_file(path_or_fileobj="app.py", path_in_repo="app.py", repo_id=repo_id, repo_type="model")

print("Uploading README...")
upload_file(path_or_fileobj="README_HF.md", path_in_repo="README.md", repo_id=repo_id, repo_type="model")

print(f"DONE! Check https://huggingface.co/{repo_id}")
