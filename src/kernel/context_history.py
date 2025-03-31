from kernel.config import config


###! Context History
class context_history:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.blocks: list[str] = []
        self.counts: list[int] = []
        self.total_size = len(system_prompt)

    def add(self, content: str):
        while self.total_size > config.N_CTX:
            self.blocks.pop(0)
            self.total_size -= self.counts.pop(0)

        self.blocks.append(content)
        self.counts.append(len(content))
        self.total_size += len(content)

    def get(self) -> str:
        return "\n".join([self.system_prompt] + self.blocks)
