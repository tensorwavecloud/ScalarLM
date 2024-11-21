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

    async def health(self):
        return await self.engine.health()

    async def generate(self, prompts):
        return await self.engine.generate(prompts)

    async def learn_docs(self, docs):
        assert False, "Not implemented yet."

    async def learn_database(self, db):
        assert False, "Not implemented yet."

    async def learn_code(self, vcs):
        assert False, "Not implemented yet."
