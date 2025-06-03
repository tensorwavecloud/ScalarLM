from benchmark.pytorch.memcpy import benchmark_memcpy
from benchmark.pytorch.memcpy_peer import benchmark_memcpy_peer
from benchmark.pytorch.gemm import benchmark_gemm
from benchmark.pytorch.forward import benchmark_forward
from benchmark.pytorch.backward import benchmark_backward

from benchmark.roofline.plot_roofline import plot_roofline
from benchmark.roofline.plot_bandwidth_sweep import plot_bandwidth_sweep

import os

import logging

def main():
    setup_logging()

    os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

    #benchmark_memcpy()
    #benchmark_memcpy_peer()
    benchmark_gemm()
    #benchmark_forward()
    #benchmark_backward()

    plot_roofline()
    plot_bandwidth_sweep()


def setup_logging():
    logging.basicConfig(level=logging.INFO)

main()
