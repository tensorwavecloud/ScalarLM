
from mpi4py import MPI


def is_main_rank():
    return MPI.COMM_WORLD.Get_rank() == 0


def barrier():
    MPI.COMM_WORLD.Barrier()


def main_rank_only(func):
    def wrap_function(*args, **kwargs):
        result = None
        barrier()
        if is_main_rank():
            result = func(*args, **kwargs)
        barrier()
        return result

    return wrap_function
