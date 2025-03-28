import socket
import os
import torch

from cray_infra.util.get_config import get_config

slurm_config_path = "/app/cray/infra/slurm_configs/slurm.conf"
gres_config_path = "/app/cray/infra/slurm_configs/gres.conf"


def discover_clusters():
    node_info = get_node_info()

    save_node_info(node_info)

    cluster_info = get_cluster_info(node_info)

    save_cluster_info(cluster_info)


def get_node_info():
    hostname = get_hostname()
    cpu_count = get_cpu_count()
    gpu_count = get_gpu_count()

    return {"hostname": hostname, "cpu_count": cpu_count, "gpu_count": gpu_count}


def get_hostname():
    return socket.gethostname()


def get_cpu_count():
    return os.cpu_count()


def get_gpu_count():
    gpu_count = 0
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
    return gpu_count


def save_node_info(node_info):
    pass


def get_cluster_info(node_info):
    return {
        "controller_info": node_info,
        "all_nodes": [node_info],
        "partitions": [{"name": "short", "nodes": [node_info]}],
    }


def save_cluster_info(cluster_info):
    write_slurm_config(cluster_info)
    write_gres_config(cluster_info)


def write_slurm_config(cluster_info):
    node_info = cluster_info["controller_info"]

    slurm_conf_values = load_slurm_conf_values()

    slurm_conf_values["SlurmctldHost"] = node_info["hostname"]

    if node_info["gpu_count"] > 0:
        slurm_conf_values["GresTypes"] = "gpu"
    else:
        if "GresTypes" in slurm_conf_values:
            del slurm_conf_values["GresTypes"]

    save_slurm_conf_values(slurm_conf_values)

    for node in cluster_info["all_nodes"]:
        write_node_config(node)

    for partition in cluster_info["partitions"]:
        write_partition_config(partition)


def load_slurm_conf_values():
    slurm_conf_values = {}
    with open(slurm_config_path, "r") as f:
        for line in f:
            # Skip comments
            if line.strip().startswith("#"):
                continue

            # Skip lines without "="
            if "=" not in line:
                continue

            key_and_value = line.split("=")

            if len(key_and_value) != 2:
                continue

            key, value = key_and_value[0], key_and_value[1]
            slurm_conf_values[key] = value.strip()
    return slurm_conf_values


def save_slurm_conf_values(slurm_conf_values):
    with open(slurm_config_path, "w") as f:
        for key, value in slurm_conf_values.items():
            f.write(f"{key}={value}\n")


def write_node_config(node):
    """
    NodeName=hostname CPUs=64 Gres=gpu:6 State=UNKNOWN
    """
    gres_string = f"Gres=gpu:{node['gpu_count']}" if node["gpu_count"] > 0 else ""
    node_config = f"NodeName={node['hostname']} CPUs={node['cpu_count']} {gres_string} State=UNKNOWN"
    with open(slurm_config_path, "a") as f:
        f.write(node_config + "\n")


def write_partition_config(partition):
    """
    PartitionName=short Nodes=node1,node2,node3 Default=YES MaxTime=INFINITE State=UP
    """

    config = get_config()

    max_training_time = (
        config["max_train_time"] + config["extra_training_seconds"]
    ) // 60

    node_names = ",".join([node["hostname"] for node in partition["nodes"]])
    partition_config = f"PartitionName={partition['name']} Nodes={node_names} Default=YES MaxTime={max_training_time} State=UP"
    with open(slurm_config_path, "a") as f:
        f.write(partition_config + "\n")


def write_gres_config(cluster_info):
    """
    NodeName=41ad10a2cba0 Name=gpu File=/dev/nvidia0
    """
    # Clear the file
    open(gres_config_path, "w").close()

    for node in cluster_info["all_nodes"]:
        gres_config = ""
        for index in get_gpu_indexes():
            if torch.version.hip:
                gres_config += (
                    f"NodeName={node['hostname']} Name=gpu File=/dev/dri/card{index}\n"
                )
            else:
                gres_config += (
                    f"NodeName={node['hostname']} Name=gpu File=/dev/nvidia{index}\n"
                )

        with open(gres_config_path, "a") as f:
            f.write(gres_config)


def get_gpu_indexes():
    # handle the case where the card is an arbtirary number
    if torch.version.hip:
        prefix = "/dev/dri"
        card_name = "card"
    else:
        prefix = "/dev"
        card_name = "nvidia"

    indexes = []

    for file in os.listdir(prefix):
        if file.startswith(card_name):
            try:
                index_as_int = int(file[len(card_name) :])
            except Exception as e:
                continue

            indexes.append(index_as_int)

    return indexes


discover_clusters()
