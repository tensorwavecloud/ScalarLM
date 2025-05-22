import os
from setuptools import setup
from torch.utils import cpp_extension

def detect_gpu_platform():
    """Determine GPU platform through environment variables and path checks"""
    if os.path.exists('/opt/rocm'):
        return 'rocm'
    if os.path.exists('/usr/local/cuda'):
        return 'cuda'
    return 'cpu'

platform = detect_gpu_platform()

include_dirs = []
library_dirs = []
compile_defines = []
libraries = []
extra_link_args = []

if platform == 'rocm':
    os.environ['CXX'] = '/opt/ompi-rocm/bin/mpicxx'
    compile_defines.append(('USE_ROCM', '1'))
    include_dirs.extend([
        '/opt/rocm/include',
        '/opt/ompi-rocm/include'
    ])
    library_dirs.extend([
        '/opt/rocm/lib',
        '/opt/ompi-rocm/lib'
    ])
    extra_link_args.append('-lmpi')
elif platform == 'cuda':
    compile_defines.append(('USE_CUDA', '1'))
    include_dirs.extend([
        '/usr/local/cuda/include',
        '/opt/hpcx/ompi/include'
    ])
    library_dirs.extend([
        '/usr/local/cuda/lib64',
        '/opt/hpcx/ompi/lib'
    ])
    libraries.append('cudart')
elif platform == 'cpu':
    os.environ['CXX'] = '/usr/bin/mpicxx'
    include_dirs.extend([
        '/usr/lib/aarch64-linux-gnu/openmpi/include',
    ])
    library_dirs.extend([
        '/usr/lib/aarch64-linux-gnu/openmpi/lib',
    ])
    libraries.append('mpi')

setup(
    name="gpu_aware_mpi",
    ext_modules=[cpp_extension.CppExtension(
        'gpu_aware_mpi',
        sources=['infra/cray_infra/training/gpu_aware_mpi/gpu_aware_mpi.cpp'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args={
            'cxx': [f'-D{name}={value}' for name, value in compile_defines]
        },
        extra_link_args=extra_link_args,
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
