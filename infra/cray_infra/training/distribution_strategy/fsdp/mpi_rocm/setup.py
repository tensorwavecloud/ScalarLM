from setuptools import setup
from torch.utils import cpp_extension

setup(name="mpi_rocm",
        ext_modules=[cpp_extension.CppExtension(
            'mpi_rocm', ['mpi_rocm.cpp'], 
            include_dirs=['/opt/rocm/include', '/opt/ompi-rocm/include'], 
            library_dirs=['/opt/ompi-rocm/lib'], 
            libraries=['mpi'], 
            extra_compile_args=['-DUSE_ROCM'])],
        cmdclass={'build_ext': cpp_extension.BuildExtension})