
from gpu_aware_mpi import get_rank, barrier


def is_main_rank():
    return get_rank() == 0

def main_rank_only(func):
    def wrap_function(*args, **kwargs):
        result = None
        barrier()
        if is_main_rank():
            result = func(*args, **kwargs)
        barrier()
        return result

    return wrap_function
