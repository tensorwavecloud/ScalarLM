from cray_infra.slurm.discovery.discover_clusters import load_all_nodes

def get_node_count():
    node_info = load_all_nodes()

    return len(node_info)
