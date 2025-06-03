from benchmark.roofline.plot_roofline import get_machine_roofline, get_processor_name

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from textwrap import wrap
import humanize

import json
import os
import math


def plot_bandwidth_sweep():
    machine_roofline = get_machine_roofline()

    plot_memcpy_sweep(machine_roofline)
    plot_memcpy_peer_sweep(machine_roofline)


def plot_memcpy_sweep(machine_roofline):
    memcpy_metrics = load_memcpy_metrics()

    plot_sweep("Memcpy", machine_roofline, memcpy_metrics)


def load_memcpy_metrics():
    metrics_path = "/app/cray/data/benchmark_memcpy.json"

    if not os.path.exists(metrics_path):
        return None

    with open(metrics_path, "r") as metrics_file:
        benchmark_metrics = json.load(metrics_file)

    return benchmark_metrics


def plot_memcpy_peer_sweep(machine_roofline):
    memcpy_peer_metrics = load_memcpy_peer_metrics()

    machine_roofline["memory_bandwidth"] = machine_roofline["peer_bandwidth"]

    plot_sweep("MemcpyPeer", machine_roofline, memcpy_peer_metrics)


def load_memcpy_peer_metrics():
    metrics_path = "/app/cray/data/benchmark_memcpy_peer.json"

    if not os.path.exists(metrics_path):
        return None

    with open(metrics_path, "r") as metrics_file:
        benchmark_metrics = json.load(metrics_file)

    return benchmark_metrics


def plot_sweep(kernel_type, machine_roofline, kernel_metrics):
    if kernel_metrics is None:
        return

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Data Size (MB)")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title(f"{get_processor_name()} {kernel_type} Bandwidth ")

    # Colorblind-friendly palette (blue, orange, teal, red, purple)
    colors = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC"]

    # Different marker shapes for better distinction
    markers = ["o", "s", "^", "D", "v", "p", "*", "X", "h"]

    max_data_size = 2**20  # 1 GB
    min_bandwidth = machine_roofline["memory_bandwidth"]

    # Track data points for later annotation
    points_data = []

    # Plot kernel data points with distinct markers and colors
    for i, (kernel_name, kernel_metrics) in enumerate(kernel_metrics.items()):
        bandwidth = kernel_metrics["bandwidth"]
        size = kernel_metrics["bytes"]

        # Use modulo to cycle through colors and markers if more kernels than options
        color_idx = i % len(colors)
        marker_idx = i % len(markers)

        # Plot with larger, more distinct markers
        x_val = size / (2**20)
        y_val = bandwidth / 1e9

        ax.plot(
            x_val,
            y_val,
            marker=markers[marker_idx],
            markersize=8,
            markerfacecolor=colors[color_idx],
            markeredgecolor="black",
            markeredgewidth=1,
            linestyle="none",  # No line connecting points
            label="\n".join(wrap(humanize.naturalsize(int(kernel_name)), 10)),
        )

        # Store point data for annotation
        points_data.append((x_val, y_val, kernel_name))

        max_data_size = max(max_data_size, size)
        min_bandwidth = min(min_bandwidth, bandwidth)

    # Plot roofline with distinct pattern
    ax.plot(
        [0.01, max_data_size * 1.1 / (2**20)],
        [
            machine_roofline["memory_bandwidth"] / 1e9,
            machine_roofline["memory_bandwidth"] / 1e9,
        ],
        linestyle="--",
        linewidth=2.5,
        color="#000000",  # Black for maximum contrast
        label="Roofline",
    )

    # Add a text label directly on the roofline
    roofline_y = machine_roofline["memory_bandwidth"] / 1e9
    ax.text(
        math.exp(math.log(max_data_size / (2**20)) * 0.3),
        machine_roofline["memory_bandwidth"] / 1e9 * 1.5,
        f"Roofline: {roofline_y:.1f} GB/s",
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
    )

    # Set axis limits
    ax.set_xlim(0.1, max_data_size * 1.1 / (2**20))
    ax.set_ylim(
        math.exp(math.log(min_bandwidth / (1e9)) * 0.9),
        machine_roofline["memory_bandwidth"] / 1e9 * 3,
    )

    # Add gridlines for easier reading of values
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    # Shrink current axis and position legend better
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Create more accessible legend with larger font
    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=True,  # Add frame
        framealpha=0.9,  # Make frame opaque
        edgecolor="gray",  # Frame color
        #        fontsize='medium'   # Increase font size
    )

    # Change the formatting of the axis tick labels
    formatter = ScalarFormatter()
    formatter.set_scientific(False)  # Turn off scientific notation
    formatter.set_useOffset(False)  # Don't use offset

    # Apply the formatter to both axes
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Save the figure
    save_to_file(fig, f"bandwidth_sweep_{kernel_type}.pdf")


def save_to_file(fig, filename):
    directory = "/app/cray/data"

    full_path = os.path.join(directory, filename)

    fig.savefig(full_path)
