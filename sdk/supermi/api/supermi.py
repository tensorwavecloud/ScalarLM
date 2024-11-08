
class SuperMI:
    def __init__(self):
        self.engine = CrayEngine()

    def train(self, data):
        return self.engine.train(data)

    def generate(self, prompts):
        return self.engine.generate(prompts)

    def learn_docs(self, docs):
        assert False, "Not implemented yet."

    def learn_database(self, db):
        assert False, "Not implemented yet."

    def learn_code(self, vcs):
        assert False, "Not implemented yet."



