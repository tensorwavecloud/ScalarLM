from cray_infra.util.get_config import get_config

import torch

from atomicwrites import atomic_write

import subprocess
import time
import socket
import os
import json

import logging

logger = logging.getLogger(__name__)

slurm_config_path = "/app/cray/infra/slurm_configs/slurm.conf"
cluster_info_path = "/app/cray/infra/slurm_configs/cluster_info.json"
cgroup_config_path = "/app/cray/infra/slurm_configs/cgroup.conf"

shared_slurm_config_path = "/app/cray/nfs/slurm.conf"
shared_gres_config_path = "/app/cray/nfs/gres.conf"
shared_cgroup_config_path = "/app/cray/nfs/cgroup.conf"
shared_node_config_directory = "/app/cray/nfs/nodes"


def main():
    setup_logging()
    discover_clusters()


def discover_clusters():

    clean_old_node_info()

    node_info = get_node_info()

    save_node_info(node_info)

    cluster_info = get_cluster_info(node_info)

    save_cluster_info(cluster_info)
    reload_slurm_configs()


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Logging setup complete.")


def clean_old_node_info():
    config = get_config()

    time_limit = config["node_info_time_limit"]

    current_time = time.time()

    if not os.path.exists(shared_node_config_directory):
        return

    for filename in os.listdir(shared_node_config_directory):
        file_path = os.path.join(shared_node_config_directory, filename)
        if os.path.isfile(file_path):
            file_mtime = os.path.getmtime(file_path)
            if current_time - file_mtime > time_limit:
                logging.debug(f"Removing old node info file: {file_path}")
                os.remove(file_path)


def get_node_info():
    hostname = get_hostname()
    cpu_count = get_cpu_count()
    gpu_count = get_gpu_count()

    return {"hostname": hostname, "cpu_count": cpu_count, "gpu_count": gpu_count, "gpu_type" : get_gpu_type(), "gpu_indexes" : get_gpu_indexes() }


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
    node_config_path = os.path.join(
        shared_node_config_directory, f"{node_info['hostname']}.json"
    )

    os.makedirs(shared_node_config_directory, exist_ok=True)

    with open(node_config_path, "w") as f:
        json.dump(node_info, f, indent=4)


def get_cluster_info(node_info):

    all_nodes = load_all_nodes()

    controller_info = elect_controller(all_nodes)

    return {
        "controller_info": controller_info,
        "all_nodes": all_nodes,
        "partitions": [{"name": "short", "nodes": all_nodes}],
    }


def load_all_nodes():
    all_nodes = []
    for filename in os.listdir(shared_node_config_directory):
        if filename.endswith(".json"):
            file_path = os.path.join(shared_node_config_directory, filename)
            with open(file_path, "r") as f:
                node_info = json.load(f)
                all_nodes.append(node_info)
    return all_nodes


def elect_controller(all_nodes):
    """
    Elects the controller node based on the lowest GPU count.
    If multiple nodes have the same CPU count, alphabetical order of hostname is used.
    """
    if not all_nodes:
        raise ValueError("No nodes found in the cluster.")

    # Sort nodes by GPU count (ascending) and hostname (alphabetical)
    all_nodes.sort(key=lambda x: (x["gpu_count"], x["hostname"]))

    # The first node in the sorted list is the controller
    controller_node = all_nodes[0]

    logger.info(
        f"Controller node elected: {controller_node['hostname']} with {controller_node['gpu_count']} GPUs"
    )

    return controller_node


def is_controller(node_info, controller_info):
    is_controller = node_info["hostname"] == controller_info["hostname"]
    return is_controller


def save_cluster_info(cluster_info):
    old_cluster_info = load_cluster_info_file()

    if old_cluster_info:
        if old_cluster_info == cluster_info:
            logger.info("Cluster info is unchanged, skipping write.")
            return

    write_slurm_config(cluster_info)
    write_gres_config(cluster_info)
    write_cgroup_config(cluster_info)
    write_cluster_info_file(cluster_info)


def load_cluster_info_file():
    if not os.path.exists(cluster_info_path):
        return None

    with open(cluster_info_path, "r") as f:
        try:
            cluster_info = json.load(f)
            return cluster_info
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {cluster_info_path}: {e}")
            return None


def write_slurm_config(cluster_info):
    node_info = cluster_info["controller_info"]

    slurm_conf_values = load_slurm_conf_values()

    slurm_conf_values["SlurmctldHost"] = node_info["hostname"]

    if has_any_gpus(cluster_info):
        slurm_conf_values["GresTypes"] = "gpu"
    else:
        if "GresTypes" in slurm_conf_values:
            del slurm_conf_values["GresTypes"]

    new_config = save_slurm_conf_values(slurm_conf_values)

    for node in cluster_info["all_nodes"]:
        new_config += write_node_config(node)

    for partition in cluster_info["partitions"]:
        new_config += write_partition_config(partition)

    with atomic_write(shared_slurm_config_path, overwrite=True) as f:
        f.write(new_config)



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

def has_any_gpus(cluster_info):
    for node in cluster_info["all_nodes"]:
        if node["gpu_count"] > 0:
            return True
    return False


def save_slurm_conf_values(slurm_conf_values):
    config = ""
    with open(shared_slurm_config_path, "w") as f:
        for key, value in slurm_conf_values.items():
            config += f"{key}={value}\n"

    return config

def write_node_config(node):
    """
    NodeName=hostname CPUs=64 Gres=gpu:6 State=UNKNOWN
    """
    gres_string = f"Gres=gpu:{node['gpu_count']}" if node["gpu_count"] > 0 else ""
    node_config = f"NodeName={node['hostname']} CPUs={node['cpu_count']} {gres_string} State=UNKNOWN"
    return node_config + "\n"


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

    return partition_config + "\n"


def write_gres_config(cluster_info):
    """
    NodeName=41ad10a2cba0 Name=gpu File=/dev/nvidia0
    """
    gres_config = ""
    for node in cluster_info["all_nodes"]:
        for index in node["gpu_indexes"]:
            if node["gpu_type"] == "amd":
                gres_config += (
                    f"NodeName={node['hostname']} Name=gpu File=/dev/dri/card{index}\n"
                )
            else:
                gres_config += (
                    f"NodeName={node['hostname']} Name=gpu File=/dev/nvidia{index}\n"
                )

    with atomic_write(shared_gres_config_path, overwrite=True) as f:
        f.write(gres_config)

def write_cgroup_config(cluster_info):
    with open(cgroup_config_path) as config:
        with atomic_write(shared_cgroup_config_path, overwrite=True) as shared_config:
            shared_config.write(config.read())


def get_gpu_indexes():
    # handle the case where the card is an arbtirary number
    if torch.version.hip:
        prefix = "/dev/dri"
        card_name = "card"
    else:
        prefix = "/dev"
        card_name = "nvidia"

    indexes = []

    if os.path.exists(prefix):
        for file in os.listdir(prefix):
            if file.startswith(card_name):
                try:
                    index_str = file[len(card_name) :]
                    if index_str.isdigit():
                        print(file[len(card_name) :])
                        index_as_int = int(file[len(card_name) :])
                        indexes.append(index_as_int)
                except Exception as e:
                    continue

    return indexes

def get_gpu_type():
    if torch.version.hip:
        return "amd"
    else:
        return "nvidia"

    return "none"


def write_cluster_info_file(cluster_info):
    with open(cluster_info_path, "w") as f:
        json.dump(cluster_info, f, indent=4)
    logger.info(f"Cluster info saved to {cluster_info_path}")


def reload_slurm_configs():
    try:
        subprocess.run(["scontrol", "reconfigure"], check=True)
        logger.info("Slurm configurations reloaded successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to reload Slurm configurations: {e}")


if __name__ == "__main__":
    main()
