from masint.engines.async_cray import AsyncCray


class AsyncSupermassiveIntelligence:
    def __init__(self):
        self.engine = AsyncCray()

    async def train(self, data, model_name=None, train_args={}):
        if model_name is not None:
            train_args["llm_name"] = model_name

        return await self.engine.train(
            data=data, model_name=model_name, train_args=train_args
        )

    async def generate(self, prompts, model_name=None, max_tokens=None):
        return await self.engine.generate(
            prompts=prompts, model_name=model_name, max_tokens=max_tokens
        )

    async def submit_generate(self, prompts, model_name=None, max_tokens=None):
        return await self.engine.submit_generate(
            prompts=prompts, model_name=model_name, max_tokens=max_tokens
        )

    async def embed(self, prompts, model_name=None):
        return await self.engine.embed(prompts=prompts, model_name=model_name)

    async def learn_classes(self, classes):
        assert False, "Not implemented yet."

    async def learn_docs(self, docs):
        assert False, "Not implemented yet."

    async def learn_database(self, db):
        assert False, "Not implemented yet."

    async def learn_code(self, vcs):
        assert False, "Not implemented yet."

    async def list_models(self):
        return await self.engine.list_models()

    async def get_training_job(self, job_dir):
        return await self.engine.get_training_job(job_dir=job_dir)

    async def health(self):
        return await self.engine.health()

    async def get_results(self, request_ids):
        return await self.engine.get_results(request_ids)
