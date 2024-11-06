import shutil
from pathlib import PurePosixPath
from typing import Union
import modal
from common import rmdir

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

app = modal.App(
    "example-megatron",
    secrets=[
        modal.Secret.from_name("HF_TOKEN"),
        modal.Secret.from_name("wandb"),
    ],
)


# Volumes for pre-trained models and training runs.
hf_models_volume = modal.Volume.from_name("hf_models_volume", create_if_missing=True)
mcore_models_volume = modal.Volume.from_name("mcore_models_volume", create_if_missing=True)
runs_volume = modal.Volume.from_name("run_volume", create_if_missing=True)

hf_models_mount = "/hf_models"
mcore_models_mount = "/mcore_models"
runs_mount = "/runs"

VOLUME_CONFIG: dict[Union[str, PurePosixPath], modal.Volume] = {
    hf_models_mount: hf_models_volume,
    mcore_models_mount: mcore_models_volume,
    runs_mount: runs_volume,
}

@app.function(
    image=modal.Image.debian_slim(python_version="3.11")
        .pip_install(["huggingface_hub==0.26.2"]),
    timeout=30 * MINUTES,
    volumes=VOLUME_CONFIG)
def hf_to_mcore():
    from huggingface_hub import snapshot_download

    repo_id = "mistralai/Mixtral-8x7B-v0.1"
    local_dir = f"/{hf_models_mount}/{repo_id}"
    local_dir_tmp = f"/{local_dir}.tmp"

    print(local_dir)
    rmdir(local_dir_tmp)
    try:
        snapshot_download(repo_id=repo_id,  ignore_patterns=["*.pt"], local_dir=local_dir, local_files_only=True)
        print(f"Volume contains {repo_id} ...")
    except FileNotFoundError:
        print(f"Downloading {repo_id} ...")
        snapshot_download(repo_id=repo_id, ignore_patterns=["*.pt"], local_dir=local_dir_tmp)
        print("Committing...")
        hf_models_volume.commit()
        shutil.move(local_dir_tmp, local_dir)

@app.local_entrypoint()
def main():
    # Wait for the training run to finish.
    hf_to_mcore.remote()
    print("Done")