from application import LazyAgentApplicationStates, LazyAgentApplicationState

from modules import LazyAgentModuleManager
from module_llm import LazyAgentModuleLLM
from module_tui import LazyAgentModuleTUI


class LazyAgentCore:
    def __init__(self) -> None:
        LazyAgentApplicationState.set_state(LazyAgentApplicationStates.CORE_RUNNING)

        self.module_manager = LazyAgentModuleManager()

        self.module_manager.register_module(LazyAgentModuleTUI())
        self.module_manager.register_module(LazyAgentModuleLLM())

    def run(self) -> None:
        while LazyAgentApplicationState.is_running():
            self.module_manager.run_module("TUI")
            self.module_manager.run_module("LLM")

    @staticmethod
    def launch() -> None:
        """
        Launch the LazyAgent core.
        """
        try:
            agent = LazyAgentCore()
            agent.run()
        except Exception as e:
            pass  # TODO: Log the exception properly
