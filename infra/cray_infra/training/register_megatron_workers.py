from cray_infra.slurm.discovery.discover_clusters import discover_clusters

async def register_megatron_workers():
    discover_clusters()
