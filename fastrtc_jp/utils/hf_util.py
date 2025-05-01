import sys,os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


def download_hf_hub(repo_id: str, path: str|None=None, *, subfolder:str|None=None, cache_dir: str|Path|None=None) -> str:
    # Hugging Face Hubからモデルをダウンロード
    for b in (True, False):
        try:
            if path:
                model_path = hf_hub_download(
                    repo_id=repo_id, filename=path,
                    subfolder=subfolder, cache_dir=cache_dir,
                    local_files_only=b,
                )
            else:
                model_path = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=cache_dir, local_files_only=b,
                )
            return model_path
        except LocalEntryNotFoundError as e:
            pass
        except Exception as e:
            raise e
    raise FileNotFoundError(f"{repo_id} {subfolder} {path} not found")
