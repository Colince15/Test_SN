# scripts/deploy.py

import os
from huggingface_hub import HfApi, HfFolder
import json

# Load configuration
with open('config.json') as f:
    config = json.load(f)

# Get secrets from environment variables
hf_username = os.getenv('HF_USERNAME')
hf_token = os.getenv('HF_TOKEN')

if not hf_username or not hf_token:
    raise ValueError("Hugging Face username and token must be set as environment variables (HF_USERNAME, HF_TOKEN)")

# Authenticate with Hugging Face
HfFolder.save_token(hf_token)

# Initialize the HfApi
api = HfApi()
repo_id = f"{hf_username}/{config['hf_repo_name']}"

# Create the repository on Hugging Face Hub if it doesn't exist
api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
print(f"Repo {repo_id} created or already exists.")

# Upload the entire project folder
print(f"Uploading files to {repo_id}...")
api.upload_folder(
    folder_path=".",  # Upload everything in the current directory
    repo_id=repo_id,
    repo_type="model",
    commit_message="Automatic deployment from GitHub Actions"
)

print("Deployment to Hugging Face Hub successful!")