import enum


# Commands
class CliLlmCommand:
    def __init__(self, name: str):
        self.name = name

    def execute(self, args: list[str]) -> str:
        # Ignore
        del args

        raise NotImplementedError("execute hasn't been implemented yet.")

    def get_name(self) -> str:
        return self.name


# Modes
class CliLlmMode:
    # Commands
    def execute_command(self, cmd: str, args: list[str]) -> str:
        # Ignore
        del cmd
        del args

        raise NotImplementedError(
            "Mode `self.execute_command` hasn't been implemented yet."
        )

    def commands(self) -> list[CliLlmCommand]:
        raise NotImplementedError("Mode `self.commands` hasn't been implemented yet.")

    # Prompt
    def compute_mode_prompt(self) -> str:
        raise NotImplementedError(
            "Mode `self.compute_mode_prompt` hasn't been implemented yet."
        )


class CliLlmModeShell(CliLlmMode):
    # TODO: Mode Shell
    # TODO: Shell allowed commands (standard bash commands)
    pass


class CliLlmModeDebugger(CliLlmMode):
    # TODO: Mode Debugger
    # TODO: Debugger commands (all gdb, pydbg, etc...)
    # TODO: Debugger open and close commands (switch mode)
    pass


class CliLlmModeFileIO(CliLlmMode):
    # TODO: Mode FileIO
    # TODO: File IO commands (custom commands format)
    # TODO: File IO open and close commands (switch mode)
    # TODO: Allowed directories to read, write.
    pass


class CliLlmModeMemory(CliLlmMode):
    # TODO: Mode Memory
    # TODO: Custom commands (memorize, recall, details...). To be improved
    pass


class CliLlmModeWebSearch(CliLlmMode):
    # TODO: Mode Web Search
    # TODO: Commands based on web search API
    pass


class CliLlmModeTaskManagement(CliLlmMode):
    # TODO: Mode Task Management
    # TODO: Task add and update, subtasks, tasks in a specific mode...
    pass


class CliLlmModeUserInteract(CliLlmMode):
    # TODO: Mode to ask the user something (blocking & non-blocking behaviors)
    pass


class CliLlmModeGit(CliLlmMode):
    # TODO: Mode GIT
    # TODO: All git commands (plus some parameters such as branches that can be modified etc...)
    pass


class CliLlmModes(enum.Enum):
    SHELL = CliLlmModeShell()
    DEBUGGER = CliLlmModeDebugger()
    FILE_IO = CliLlmModeFileIO()
    MEMORY = CliLlmModeMemory()
    WEB_SEARCH = CliLlmModeWebSearch()
    TASK_MANAGEMENT = CliLlmModeTaskManagement()  # NOTE: Default starting mode!
    USER_INTERACT = CliLlmModeUserInteract()
    GIT = CliLlmModeGit()


# Config
class CliLlmConfig:
    MODEL_NAME = "qwen3"
    MODEL_QUANT = "qwen3..."
    PROVIDER = "Groq<Down>"
    ALLOWED_MODES = []
    PROJECT_DIRECTORY = ""
    TOP_P = 0.0
    TOP_K = 0.0
    MIN_P = 0.0
    MIN_K = 0.0
    TEMPERATURE = 0.0
    SIZE_CTX = 8192
    SIZE_BATCH = 512


# History
class CliLlmShortTermMemory:
    # TODO: History
    pass


class CLiLlmVectorialMemory:
    # TODO: Vectorial Memory
    pass


# Prompting
class CliLlmPromptFormat:
    # System
    def get_system_tag_beg(self) -> str:
        raise NotImplementedError("Not implemented `self.get_system_tag_beg()`!")

    def get_system_tag_end(self) -> str:
        raise NotImplementedError("Not implemented `self.get_system_tag_end()`!")

    # User
    def get_user_tag_beg(self) -> str:
        raise NotImplementedError("Not implemented `self.get_user_tag_beg()`!")

    def get_user_tag_end(self) -> str:
        raise NotImplementedError("Not implemented `self.get_user_tag_end()`!")

    # Assistant
    def get_assistant_tag_beg(self) -> str:
        raise NotImplementedError("Not implemented `self.get_assistant_tag_beg()`!")

    def get_assistant_tag_end(self) -> str:
        raise NotImplementedError("Not implemented `self.get_assistant_tag_end()`!")


class CliLlmUserCLI:
    # TODO:
    pass


class CliLlmUserPrompter:
    # TODO: Prompter
    pass


# Interpreter
class CliLlmResultInterpreter:
    # TODO: Result Interpreter
    pass


# Engine
class CliLlmEngineProvider:
    # TODO: Engine Provider

    def produce(self, prompt: str) -> str:
        # Ignore
        del prompt

        raise NotImplementedError(
            "Engine Provider `produce` hasn't been implemented yet."
        )


class CliLlmEngineProviderGroq(CliLlmEngineProvider):
    # TODO: Engine Provider -> Groq
    pass


class CliLlmEngineProviderLlama(CliLlmEngineProvider):
    # TODO: Engine Provider -> Llama
    pass


class CliLlmEngineProviderOpenAI(CliLlmEngineProvider):
    # TODO: Engine Provider -> OpenAI
    pass


class CliLlmModule:
    def run(self):
        raise NotImplementedError("Module hasn't been implemented yet.")


class CliLlmPromptEngine(CliLlmModule):
    # Running
    def run(self):
        pass


class CliLlmApplication:
    def __init__(self):
        self.prompt_engine = CliLlmPromptEngine()

    def run(self):
        self.prompt_engine.run()


# Main
if __name__ == "__main__":
    app = CliLlmApplication()
    app.run()
