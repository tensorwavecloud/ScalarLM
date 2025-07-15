from cray_infra.slurm.discovery.discover_clusters import load_all_nodes

def get_gpu_count():
    node_info = load_all_nodes()

    gpu_count = 0

    for node in node_info:
        gpu_count = max(node["gpu_count"], gpu_count)

    return gpu_count
