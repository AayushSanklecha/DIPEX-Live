import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

api = HfApi(token=os.environ.get("HF_API_KEY"))

print("Uploading dashboard...")
try:
    api.upload_folder(
        folder_path="dashboard",
        path_in_repo="dashboard",
        repo_id="AayuSanklu/DIPEX-Live",
        repo_type="space"
    )
    print("Dashboard uploaded successfully!")
except Exception as e:
    print(f"Failed: {e}")
