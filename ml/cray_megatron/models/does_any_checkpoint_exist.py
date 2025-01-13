from cray_megatron.models.get_latest_checkpoint_path import (
    get_latest_checkpoint_path,
)


def does_any_checkpoint_exist():
    return get_latest_checkpoint_path() is not None
