from benchmark.roofline.plot_roofline import get_machine_roofline, get_processor_name

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from textwrap import wrap
import humanize

import json
import os



def plot_bandwidth_sweep():
    machine_roofline = get_machine_roofline()

    plot_memcpy_sweep(machine_roofline)
    plot_memcpy_peer_sweep(machine_roofline)


def plot_memcpy_sweep(machine_roofline):
    memcpy_metrics = load_memcpy_metrics()

    plot_sweep("Memcpy", machine_roofline, memcpy_metrics)


def load_memcpy_metrics():
    metrics_path = "/app/cray/data/benchmark_memcpy.json"

    with open(metrics_path, "r") as metrics_file:
        benchmark_metrics = json.load(metrics_file)

    return benchmark_metrics


def plot_memcpy_peer_sweep(machine_roofline):
    memcpy_peer_metrics = load_memcpy_peer_metrics()

    plot_sweep("MemcpyPeer", machine_roofline, memcpy_peer_metrics)


def load_memcpy_peer_metrics():
    metrics_path = "/app/cray/data/benchmark_memcpy_peer.json"

    with open(metrics_path, "r") as metrics_file:
        benchmark_metrics = json.load(metrics_file)

    return benchmark_metrics


def plot_sweep(kernel_type, machine_roofline, kernel_metrics):
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Data Size (MB)")

    ax.set_ylabel("Bandwidth (GB/s)")

    ax.set_title(f"{kernel_type} Bandwidth Sweep for {get_processor_name()}")

    max_data_size = 2**20  # 1 GB
    min_bandwidth = machine_roofline["memory_bandwidth"]

    for kernel_name, kernel_metrics in kernel_metrics.items():
        bandwidth = kernel_metrics["bandwidth"]
        size = kernel_metrics["bytes"]

        # plot a single point
        ax.plot(
            size / (2**20),
            bandwidth / 2**30,
            "o",
            label="\n".join(wrap(humanize.naturalsize(int(kernel_name)), 10)),
        )

        max_data_size = max(max_data_size, size)
        min_bandwidth = min(min_bandwidth, bandwidth)

    ax.plot(
        [
            0.01,
            max_data_size * 1.1 / (2**20),
        ],
        [
            machine_roofline["memory_bandwidth"] / 2**30,
            machine_roofline["memory_bandwidth"] / 2**30,
        ],
        label="Roofline",
    )

    ax.set_xlim(0.1, max_data_size * 1.1 / (2**20))
    ax.set_ylim(
        min_bandwidth * 0.9 / (2**30),
        machine_roofline["memory_bandwidth"] * 1.1 / 2**30,
    )

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Change the formatting of the axis tick labels
    formatter = ScalarFormatter()
    formatter.set_scientific(False)  # Turn off scientific notation
    formatter.set_useOffset(False)   # Don't use offset

    # Apply the formatter to both axes
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    save_to_file(fig, f"bandwidth_sweep_{kernel_type}.pdf")


def save_to_file(fig, filename):
    directory = "/app/cray/data"

    full_path = os.path.join(directory, filename)

    fig.savefig(full_path)
