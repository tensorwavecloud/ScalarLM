from cray_megatron.collectives.main_rank_only import main_rank_only

from huggingface_hub import snapshot_download


@main_rank_only
def download_model(model_name):
    snapshot_download(repo_id=model_name)
