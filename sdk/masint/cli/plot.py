from masint.util.make_api_url import make_api_url

import matplotlib.pyplot as plt

import os
import aiohttp
import asyncio

import traceback

import logging

logger = logging.getLogger(__name__)


def plot(model_name, smooth):
    logger.info(f"Plotting model {model_name}")

    try:
        asyncio.run(plot_async(model_name=model_name, smooth=smooth))
    except Exception as e:
        logger.error(f"Failed to plot model {model_name}")
        logger.error(e)
        logger.error(traceback.format_exc())


async def plot_async(model_name, smooth):
    status = await get_status(model_name)

    history = status["job_status"]["history"]
    model_name = clip_model_name(
        os.path.basename(status["job_config"]["job_directory"])
    )

    # Plot loss against step
    plot_loss(history, model_name, smooth)


def plot_loss(history, model_name, smooth):
    steps = [int(entry["step"]) for entry in history]
    losses = [float(entry["loss"]) for entry in history]

    losses = apply_smoothing(losses, smooth)

    plt.plot(steps, losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training loss for model " + model_name)
    plt.grid(True)

    base_path = "/app/cray/data"

    if os.path.isdir(base_path):
        path = f"{base_path}/loss_plot_{model_name}.pdf"
    else:
        path = f"loss_plot_{model_name}.pdf"

    logger.info(f"Saving plot to {path}")

    plt.savefig(path)


def apply_smoothing(data, smooth):
    # Number of steps to smooth over
    smooth_steps = smooth

    smoothed_data = []

    for i in range(len(data)):
        if i < smooth_steps:
            smoothed_data.append(sum(data[: i + 1]) / (i + 1))
        else:
            smoothed_data.append(sum(data[i - smooth_steps : i]) / smooth_steps)

    return smoothed_data


async def get_status(model_name):
    async with aiohttp.ClientSession() as session:
        async with session.get(make_api_url(f"v1/megatron/train/{model_name}")) as resp:
            data = await resp.json()

            logger.info(f"Got status for model {model_name}")
            logger.info(data)

            if resp.status != 200:
                logger.error(f"Failed to get status for model {model_name}")
                logger.error(data)
                raise Exception("Failed to get status for model")

            return data


def clip_model_name(model_name):
    return model_name[-10:]
