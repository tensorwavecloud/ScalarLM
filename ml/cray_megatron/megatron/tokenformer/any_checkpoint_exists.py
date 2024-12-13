from cray_megatron.megatron.tokenformer.get_latest_checkpoint_path import (
    get_latest_checkpoint_path,
)


def any_checkpoint_exists():
    return get_latest_checkpoint_path() is not None
