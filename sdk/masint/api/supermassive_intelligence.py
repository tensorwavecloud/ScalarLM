from masint.api.async_supermassive_intelligence import AsyncSupermassiveIntelligence

import asyncio


class SupermassiveIntelligence:
    def __init__(self):
        self.async_api = AsyncSupermassiveIntelligence()

    ## Train and Inference API methods

    def train(self, data, model_name=None, train_args={}):
        return asyncio.run(
            self.async_api.train(
                data=data, model_name=model_name, train_args=train_args
            )
        )

    def generate(self, prompts, model_name=None, max_tokens=None):
        return asyncio.run(
            self.async_api.generate(
                prompts=prompts, model_name=model_name, max_tokens=max_tokens
            )
        )

    def embed(self, prompts, model_name=None):
        return asyncio.run(self.async_api.embed(prompts=prompts, model_name=model_name))

    ## Alignment API methods

    def learn_classes(self, classes):
        return asyncio.run(self.async_api.learn_classes(classes=classes))

    def learn_docs(self, docs):
        return asyncio.run(self.async_api.learn_docs(docs=docs))

    def learn_database(self, db):
        return asyncio.run(self.async_api.learn_database(db=db))

    def learn_code(self, vcs):
        return asyncio.run(self.async_api.learn_code(vcs=vcs))

    ## Status API methods

    def list_models(self):
        return asyncio.run(self.async_api.list_models())

    def get_training_job(self, job_dir):
        return asyncio.run(self.async_api.get_training_job(job_dir=job_dir))

    def health(self):
        return asyncio.run(self.async_api.health())
