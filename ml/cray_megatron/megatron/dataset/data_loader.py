from cray_megatron.megatron.dataset.load_dataset import load_dataset

from cray_infra.util.get_job_config import get_job_config

import torch


class DataLoader:
    def __init__(self, model, tokenizer):

        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = get_batch_size()
        self.epoch = 0

        self.dataset = load_dataset(
            model=self.model,
            tokenizer=self.tokenizer,
            epoch=self.epoch,
        )

        self.loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size
        )

    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.dataset = load_dataset(
                model=self.model,
                tokenizer=self.tokenizer,
                epoch=self.epoch,
            )
            self.loader = torch.utils.data.DataLoader(
                self.dataset, batch_size=self.batch_size
            )
            self.iterator = iter(self.loader)

            return next(self.iterator)


def get_batch_size():
    job_config = get_job_config()
    return job_config["batch_size"]
