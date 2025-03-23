import torch
from mpi4py import MPI

def get_mpi_datatype(tensor: torch.Tensor):
    dtype = tensor.dtype
    if dtype == torch.float32:
        return MPI.FLOAT
    elif dtype == torch.float64:
        return MPI.DOUBLE
    elif dtype == torch.int32:
        return MPI.INT
    elif dtype == torch.int64:
        return MPI.LONG
    elif dtype == torch.uint8:
        return MPI.UNSIGNED_CHAR
    elif dtype == torch.int8:
        return MPI.SIGNED_CHAR
    elif dtype == torch.bool:
        return MPI.C_BOOL
    elif dtype == torch.complex64:
        return MPI.C_FLOAT_COMPLEX
    elif dtype == torch.complex128:
        return MPI.C_DOUBLE_COMPLEX
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")