import os
from setuptools import setup
from torch.utils import cpp_extension

def detect_gpu_platform():
    """Determine GPU platform through environment variables and path checks"""
    if os.path.exists('/opt/rocm'):
        return 'rocm'
    if os.path.exists('/usr/local/cuda'):
        return 'cuda'
    raise EnvironmentError("No supported GPU platform detected")

platform = detect_gpu_platform()

include_dirs = []
library_dirs = []
compile_defines = []
libraries = ['mpi']

if platform == 'rocm':
    compile_defines.append(('USE_ROCM', '1'))
    include_dirs.extend([
        '/opt/rocm/include',
        '/opt/ompi-rocm/include'
    ])
    library_dirs.extend([
        '/opt/rocm/lib',
        '/opt/ompi-rocm/lib'
    ])
elif platform == 'cuda':
    compile_defines.append(('USE_CUDA', '1')) 
    include_dirs.extend([
        '/usr/local/cuda/include',
        '/usr/local/mpi/include'
    ])
    library_dirs.extend([
        '/usr/local/cuda/lib64',
        '/usr/local/mpi/lib'
    ])
    libraries.append('cudart')

setup(
    name="gpu_aware_mpi",
    ext_modules=[cpp_extension.CppExtension(
        'gpu_aware_mpi',
        sources=['gpu_aware_mpi.cpp'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args={
            'cxx': [f'-D{name}={value}' for name, value in compile_defines]
        }
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
