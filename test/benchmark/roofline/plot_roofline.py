import matplotlib.pyplot as plt
from textwrap import wrap

import os
import platform
import subprocess
import re
import json
import torch

def plot_roofline():
    machine_roofline = get_machine_roofline()

    plot_gemm_kernels_on_roofline(machine_roofline)
    plot_memcpy_on_roofline(machine_roofline)
    plot_forward_on_roofline(machine_roofline)
    plot_backward_on_roofline(machine_roofline)


def plot_gemm_kernels_on_roofline(machine_roofline):
    gemm_kernel_metrics = load_gemm_kernel_metrics()

    plot_kernels_on_roofline("GEMM", machine_roofline, gemm_kernel_metrics)


def load_gemm_kernel_metrics():
    metrics_path = "/app/cray/data/benchmark_gemm.json"

    with open(metrics_path, "r") as metrics_file:
        benchmark_metrics = json.load(metrics_file)

    return benchmark_metrics


def plot_memcpy_on_roofline(machine_roofline):
    memcpy_metrics = load_memcpy_metrics()

    plot_kernels_on_roofline("Memcpy", machine_roofline, memcpy_metrics)


def load_memcpy_metrics():
    metrics_path = "/app/cray/data/benchmark_memcpy.json"

    with open(metrics_path, "r") as metrics_file:
        benchmark_metrics = json.load(metrics_file)

    return benchmark_metrics


def plot_forward_on_roofline(machine_roofline):
    forward_metrics = load_forward_metrics()

    plot_kernels_on_roofline("Inference", machine_roofline, forward_metrics)


def load_forward_metrics():
    metrics_path = "/app/cray/data/benchmark_forward.json"

    with open(metrics_path, "r") as metrics_file:
        benchmark_metrics = json.load(metrics_file)

    benchmark_metrics = flatten_metrics(prefix="", metrics=benchmark_metrics)

    return benchmark_metrics


def plot_backward_on_roofline(machine_roofline):
    backward_metrics = load_backward_metrics()

    plot_kernels_on_roofline("Training", machine_roofline, backward_metrics)


def load_backward_metrics():
    metrics_path = "/app/cray/data/benchmark_backward.json"

    with open(metrics_path, "r") as metrics_file:
        benchmark_metrics = json.load(metrics_file)

    benchmark_metrics = flatten_metrics(prefix="", metrics=benchmark_metrics)

    return benchmark_metrics


def flatten_metrics(prefix, metrics):
    flat_metrics = {}
    is_leaf = False
    for key, value in metrics.items():
        if isinstance(value, dict):
            flat_metrics = flatten_metrics(f"{prefix}{key}_", value)
        else:
            flat_metrics[f"{key}"] = value
            is_leaf = True

    if is_leaf:
        flat_metrics = {prefix: flat_metrics}

    return flat_metrics


def plot_kernels_on_roofline(kernel_type, machine_roofline, gemm_kernel_metrics):
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Operational Intensity (FLOP/Byte)")

    ax.set_ylabel("Performance (GFLOP/s)")

    ax.set_title(f"{kernel_type} Roofline Model for {get_processor_name()}")

    max_intensity = (
        10 * machine_roofline["flops"] / machine_roofline["memory_bandwidth"]
    )
    min_flops = machine_roofline["memory_bandwidth"]

    for kernel_name, kernel_metrics in gemm_kernel_metrics.items():
        intensity = kernel_metrics["operational_intensity"]
        flops = kernel_metrics["flop/s"]

        # plot a single point
        ax.plot(intensity, flops, "o", label="\n".join(wrap(kernel_name, 10)))

        max_intensity = max(max_intensity, intensity)
        min_flops = min(min_flops, flops)

    ax.plot(
        [
            0.01,
            machine_roofline["flops"] / machine_roofline["memory_bandwidth"],
            max_intensity * 1.1,
        ],
        [
            machine_roofline["memory_bandwidth"],
            machine_roofline["flops"],
            machine_roofline["flops"],
        ],
        label="Roofline",
    )

    ax.set_xlim(0.1, max_intensity * 1.1)
    ax.set_ylim(min_flops * 0.9, machine_roofline["flops"] * 1.1)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    save_to_file(fig, f"roofline_{kernel_type}.pdf")


def save_to_file(fig, filename):
    directory = "/app/cray/data"

    full_path = os.path.join(directory, filename)

    fig.savefig(full_path)


def get_machine_roofline():
    processor_name = get_processor_name()

    assert (
        processor_name in processor_table
    ), f"Processor {processor_name} not found in processor table"

    return processor_table[processor_name]


def get_processor_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)

    assert platform.system() == "Linux"
    command = "lscpu"
    all_info = subprocess.check_output(command, shell=True).decode().strip()
    for line in all_info.split("\n"):
        if "Apple" in line:
            return "Apple M2"
    return "UNKNOWN"


processor_table = {
    "Apple M2": {
        "memory_bandwidth": 102.4e9,
        "flops": 223.36e9,  # 4 cores * 3.49 GHz * 2 FLOPS/FMA * 8 ops/SIMD function unit
    },
    "AMD Instinct MI300X": {
        "memory_bandwidth": 5.3e12,
        "flops": 1.3e12,
    },
}
