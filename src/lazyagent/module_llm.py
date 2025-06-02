from modules import LazyAgentModuleAbstract


class LazyAgentModuleLLM(LazyAgentModuleAbstract):
    def __init__(self) -> None:
        super().__init__("LLM")

    def run(self) -> None:
        pass  # TODO: Implement the logic for the LLM module here
