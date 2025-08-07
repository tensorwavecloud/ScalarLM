from cray_infra.slurm.discovery.discover_clusters import load_all_nodes

def get_node_count():
    node_info = load_all_nodes()

    gpu_node_count = 0

    for node in node_info:
        if node["gpu_count"] > 0:
            gpu_node_count += 1

    if gpu_node_count > 0:
        return gpu_node_count

    return len(node_info)
