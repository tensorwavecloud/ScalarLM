from cray_infra.generate.metrics import get_metrics

async def metrics():
    """
    Get the current metrics.
    """
    return get_metrics().get_all_metrics()
