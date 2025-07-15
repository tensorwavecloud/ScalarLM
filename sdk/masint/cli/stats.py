from masint import SupermassiveIntelligence

import logging

logger = logging.getLogger(__name__)


def stats():
    logger.debug(f"Getting stats")

    try:
        llm = SupermassiveIntelligence()

        stats = llm.metrics()
    except Exception as e:
        logger.error(f"Failed to get stats")
        logger.error(e)
        return

    print_stats(stats)


def print_stats(stats):
    keys = stats.keys()

    max_lengths = {key: len(key) for key in keys}

    for key in keys:
        if "flop" in key:
            stats[key] = humanize_flops(stats[key])
        elif "/s" in key or "_time" in key:
            stats[key] = f"{stats[key]:.2f}"

    for key, value in stats.items():
        max_lengths[key] = max(max_lengths[key], len(str(value)))

    header = " | ".join(f"{key:<{max_lengths[key]}}" for key in keys)

    print(header)
    print("-" * len(header))
    row = " | ".join(f"{str(stats[key]):<{max_lengths[key]}}" for key in keys)
    print(row)

def humanize_flops(flops):
    """
    Convert FLOPS to a human-readable format.
    """
    if flops < 1e9:
        return f"{flops:.2f} FLOPS"
    elif flops < 1e12:
        return f"{flops / 1e9:.2f} GFLOPS"
    elif flops < 1e15:
        return f"{flops / 1e12:.2f} TFLOPS"
    elif flops < 1e18:
        return f"{flops / 1e15:.2f} PFLOPS"
    else:
        return f"{flops / 1e18:.2f} EFLOPS"
