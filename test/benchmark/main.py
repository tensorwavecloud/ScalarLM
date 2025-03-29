from benchmark.pytorch.memcpy import benchmark_memcpy
from benchmark.pytorch.memcpy_peer import benchmark_memcpy_peer
from benchmark.pytorch.gemm import benchmark_gemm

import logging

def main():
    setup_logging()

    benchmark_memcpy()
    benchmark_memcpy_peer()
    benchmark_gemm()

def setup_logging():
    logging.basicConfig(level=logging.INFO)

main()
