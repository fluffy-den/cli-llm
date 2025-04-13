from typing import Tuple, TextIO
from llama_cpp import Llama

import datetime
import os
import sys
import argparse
import shutil
import re


###! Configuration
class CliLlmModel:
    ## Model Source
    MODEL_NAME = "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF"
    # TODO: It would be good to either run a model locally, and if '-g' argument is passed
    # to the script, then we might run the model with groq clouds

    ## Model quantization
    MODEL_QUANT = "*q6_K.gguf"
    ## Context size
    N_CTX = 16384
    ## Offload layers
    N_GPU_LAYERS = 25
    ## Batch size
    N_BATCH = 512
    ## Temperature
    N_TEMPERATURE = 0.5
    ## Top K
    N_TOP_K = 30
    ## Top P
    N_TOP_P = 0.95
    ## Min P
    N_MIN_P = 0.05
    ## Model
    MODEL: Llama | None = None

    @staticmethod
    def init():
        if not CliLlmModel.MODEL:
            try:
                CliLlmModel.MODEL = Llama.from_pretrained(
                    repo_id=CliLlmModel.MODEL_NAME,
                    filename=CliLlmModel.MODEL_QUANT,
                    n_ctx=CliLlmModel.N_CTX,
                    n_gpu_layers=CliLlmModel.N_GPU_LAYERS,
                    n_batch=CliLlmModel.N_BATCH,
                    verbose=False,
                )
            except Exception as e:
                print(f"Error initializing model: {e}")
                sys.exit(1)

    @staticmethod
    def send_prompt(prompt: str, max_tokens: int = N_CTX):
        if not CliLlmModel.MODEL:
            raise ValueError("Model not initialized")

        max_tokens = min(max_tokens, CliLlmModel.N_CTX - len(prompt) - 32)

        return CliLlmModel.MODEL(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=CliLlmModel.N_TEMPERATURE,
            top_p=CliLlmModel.N_TOP_P,
            min_p=CliLlmModel.N_MIN_P,
            top_k=CliLlmModel.N_TOP_K,
            stop=["[END]"],
        )
        # return CliLlmModel.MODEL.create_completion(
        #     prompt=prompt,
        #     max_tokens=max_tokens,
        #     temperature=CliLlmModel.N_TEMPERATURE,
        #     top_p=CliLlmModel.N_TOP_P,
        #     top_k=CliLlmModel.N_TOP_K,
        #     min_p=CliLlmModel.N_MIN_P,
        #     stop=["[END]"],
        #     grammar=grammar,
        # )

    @staticmethod
    def finish():
        CliLlmModel.MODEL = None

    @staticmethod
    def tokenize(prompt: str) -> list[int]:
        return (
            CliLlmModel.MODEL.tokenize(prompt.encode("utf-8"))
            if CliLlmModel.MODEL
            else []
        )


###! Command Base
class CliLlmCommandBase:
    def __init__(self, name: str):
        self.name = name

    def action(self, args: list[str]) -> str:
        raise NotImplementedError("This method must be implemented in a subclass")

    def desc_short(self) -> str:
        raise NotImplementedError("This method must be implemented in a subclass")

    def desc(self) -> str:
        raise NotImplementedError("This method must be implemented in a subclass")


###! Module Base
class CliLlmModuleBase:
    def __init__(self, name: str, dependencies: list[str] = []):
        self.name = name
        self.dependencies = dependencies

    def retrieve_commands(self) -> list[CliLlmCommandBase]:
        raise NotImplementedError("retrieve_commands() not implemented")

    def retrieve_dependencies(self) -> list[str]:
        return self.dependencies


###! Command Manager
class CliLlmCommandManager:
    def __init__(self):
        self.commands: dict[str, CliLlmCommandBase] = {}

    def get_command_list(self) -> list[CliLlmCommandBase]:
        return list(self.commands.values())

    def has_command(self, name: str) -> bool:
        return name in self.commands

    def get_command(self, name: str) -> CliLlmCommandBase:
        if not self.has_command(name):
            raise ValueError(f"Command '{name}' not found")
        return self.commands[name]

    def register(self, commands: list[CliLlmCommandBase]):
        for cmd in commands:
            self.commands[cmd.name] = cmd

    def execute(self, cmd: str, argv: list[str]) -> str:
        if cmd not in self.commands:
            return f"Command '{cmd}' not found. Please use available commands.\n"
        result = self.commands[cmd].action(argv)
        print(result)
        return result

    def clear(self):
        self.commands = {}


###! Module Manager
class CliLlmModuleManager:
    def __init__(self):
        self.modules: dict[str, CliLlmModuleBase] = {}

    def register(self, module: CliLlmModuleBase):
        # Recursive dependency resolution
        def register_dependencies(module: CliLlmModuleBase):
            for dep in module.retrieve_dependencies():
                if dep not in self.modules:
                    # Auto-register core module if needed
                    if dep == "core":
                        register_dependencies(CliLlmModuleCore())
                    else:
                        raise ValueError(f"Unresolved dependency: {dep}")
            # Register the module if not already registered
            self.modules[module.name] = module
            # Register its commands
            CliLlmKernel.COMMANDS.register(module.retrieve_commands())

        register_dependencies(module)

    def is_registered(self, name: str) -> bool:
        return name in self.modules

    def get(self, name: str) -> CliLlmModuleBase:
        return self.modules[name]

    def get_list(self) -> list[CliLlmModuleBase]:
        return list(self.modules.values())

    def get_commands(self) -> list[CliLlmCommandBase]:
        return [
            cmd
            for module in self.modules.values()
            for cmd in module.retrieve_commands()
        ]


###! Message History
class CliLlmMessageHistory:
    def __init__(self):
        self.system_prompt: str | None = None
        self.blocks: list[str] = []
        self.counts: list[int] = []
        self.total_size = 0

    def compute_tokens(self, prompt: str) -> int:
        assert CliLlmModel.MODEL is not None
        return len(CliLlmModel.MODEL.tokenize(prompt.encode("utf-8")))

    def set_system_prompt(self, system_prompt: str):
        assert self.system_prompt is None
        self.system_prompt = system_prompt
        self.total_size += self.compute_tokens(system_prompt)

    def add(self, content: str):
        assert self.system_prompt is not None
        assert CliLlmModel.MODEL is not None
        tokens_count = self.compute_tokens(content)

        while self.total_size > CliLlmModel.N_CTX and len(self.blocks) > 0:
            self.blocks.pop(0)
            self.total_size -= self.counts.pop(0)

        self.blocks.append(content)
        self.counts.append(tokens_count)
        self.total_size += tokens_count

    def get(self) -> str:
        assert self.system_prompt is not None
        return "\n".join([self.system_prompt] + self.blocks)


###! Modules
##! Core
#! Command: /exit
class CliLlmCommandExit(CliLlmCommandBase):
    def __init__(self):
        super().__init__("exit")

    def action(self, args: list[str]) -> str:
        if CliLlmModuleCore.has_done_all_tasks():
            CliLlmKernel.SHOULD_EXIT = True
            return "Exiting request has been asked...\n"
        else:
            return "You can't exit the program, because there are still pending tasks to do! Use `/pending_task_list [END]` to identify which tasks needs to be completed!\n"

    def desc(self) -> str:
        return (
            "Exit the program. IMPORTANT: Don't use it unless you're sure that your tasks are finished, and"
            " that you can't continue further!\n"
            "USAGE: `/exit [END]`\n"
        )


#! Command: /reason (MESSAGE)
class CliLlmCommandReason(CliLlmCommandBase):
    def __init__(self):
        super().__init__("say")

    def action(self, args: list[str]) -> str:
        return "Registered a thought in history!\n"

    def desc(self) -> str:
        return (
            "Say something to the user. This command allows you to write down your thoughts to the user. Your thoughts\n"
            "should contain the idea developed, a thought test, and a conclusion, as well as a list of assumptions or\n"
            "ideas to develop. Try a thought test to check that you're correct. Add Markdown to your message to\n"
            "write code, mathematics, etc.\n"
            "The message might contains code snippets, logical formula tests, mathematicals equations to resolve, and intuition.\n"
            "USAGE: `/say <MESSAGE> [END]`\n"
            "EXAMPLE: `/say Let's suppose that P. Then maybe X implies Y... [END]`\n"
            "EXAMPLE: `/say Let's say I have X=...\n That seems correct. Wait, what if $$\\mathrm{e} = \\sum_{n=0}^{\\infty} \\dfrac{1}{n!}$$\nThis isn't correct. Let's try with...[END]\n"
        )


#! Command: /create_task -c ID1, ID2... -t message
class CliLlmCommandCreateTask(CliLlmCommandBase):
    def __init__(self):
        super().__init__("create_task")

    def action(self, args: list[str]) -> str:
        i = 0
        title = ""
        dependencies = []

        while i < len(args):
            arg = args[i]
            match arg:
                case "-c":
                    i += 1
                    while i < len(args) and args[i] != "-t":
                        if args[i].startswith("-") and args[i] != "-t":
                            return f'Invalid parameter: Can\'t use -c twice or another unknown option. Got "{args[i]}"\n'
                        if not args[i].isdigit():
                            return f"Invalid argument: {args[i]} is not a valid ID. It must be a valid integer to an existing task!\n"
                        if not CliLlmModuleCore.is_valid_task(int(args[i])):
                            if CliLlmModuleCore.has_created_any_task():
                                return f"Invalid argument: Task with ID {args[i]}. Their isn't any existing task!\n"
                            return f"Invalid argument: Task with ID {args[i]} does not exist.\n"
                        dependencies.append(int(args[i]))
                        i += 1

                    if len(dependencies) == 0:
                        return "Invalid argument: -c requires at least one ID. Theses IDs must be integers of valid tasks.\n"

                case "-t":
                    i += 1
                    b = i
                    while i < len(args) and args[i] != "-c":
                        if args[i].startswith("-") and args[i] != "-c":
                            return f'Invalid parameter: Can\'t use -t twice or another unknown option. Got "{args[i]}"\n'
                        i += 1

                    title = " ".join(args[b:i])
                    if len(title) == 0:
                        return "Invalid argument: -t requires at least one word for the title of the task!\n"

                case _:
                    return "Invalid argument: " + arg + "\n"

        # Create the task
        task_id = CliLlmModuleCore.create_task(title)
        for dep_id in dependencies:
            CliLlmModuleCore.add_dependency(task_id, dep_id)
        return f"Task created with ID {task_id}.\n"

    def desc(self) -> str:
        return (
            "Create a new task with the given title and dependencies. You can specify multiple dependencies using -c ID1, ID2...\n"
            "USAGE: `/create_task -c <ID1> <ID2> ... <IDN> -t <TASK_TITLE> [END]`\n"
            "EXAMPLE: `/create_task -c 50 45 -t My new task [END]`\n"
            "         `/create_task -t My new task without dependencies [END]`\n"
        )


#! Command: /comment ID MESSAGE
class CliLlmCommandComment(CliLlmCommandBase):
    def __init__(self):
        super().__init__("comment")

    def action(self, args: list[str]) -> str:
        if len(args) < 2:
            return "Invalid argument: <ID> and MESSAGE are required.\n"

        task_id = args[0]
        message = " ".join(args[1:])

        if not task_id.isdigit():
            return f"Invalid argument: {task_id} is not a valid ID. It must be a valid integer to an existing task!\n"

        if not CliLlmModuleCore.is_valid_task(int(task_id)):
            return f"Invalid argument: Task with ID {task_id} does not exist.\n"

        CliLlmModuleCore.TASKS[int(task_id)].history.append(message)
        return f"Comment added to task {task_id}.\n"

    def desc(self) -> str:
        return (
            "Add a comment to the specified task. You can specify the task ID and the message. Do exactly the same think as specified for the `/say` command. See its details.\n"
            "USAGE: `/comment <ID> <MESSAGE> [END]`\n"
            "EXAMPLE: `/comment 469 This is a comment [END]`\n"
        )


#! Command: /done ID1, ID2...
class CliLlmCommandDone(CliLlmCommandBase):
    def __init__(self):
        super().__init__("done")

    def action(self, args: list[str]) -> str:
        if len(args) == 0:
            return "Invalid argument: At least one <ID> is required.\n"

        for arg in args:
            if not arg.isdigit():
                return f"Invalid argument: {arg} is not a valid ID. It must be a valid integer to an existing task!\n"
            if not CliLlmModuleCore.is_valid_task(int(arg)):
                return f"Invalid argument: Task with ID {arg} does not exist.\n"
            task = CliLlmModuleCore.get_task(int(arg))
            if not task or not task.status == CliLlmTOTTaskStatus.PENDING:
                return f"Invalid argument: Task with ID {arg} is not pending.\n"

        result = "Marked the following tasks as done:\n"
        for arg in args:
            task = CliLlmModuleCore.get_task(int(arg))
            assert task is not None
            task.status = CliLlmTOTTaskStatus.SUCCESS
            result += f"- Task {arg}: {task.title}\n"
        return result

    def desc(self) -> str:
        return (
            "Mark the specified tasks as done. You can specify multiple IDs using ID1, ID2...\n"
            "USAGE: `/done <ID1> <ID2> ... <IDN> [END]`\n"
            "EXAMPLE: `/done 4 21 [END]`\n"
        )


#! Command: /fail ID1, ID2...
class CliLlmCommandFail(CliLlmCommandBase):
    def __init__(self):
        super().__init__("fail")

    def action(self, args: list[str]) -> str:
        if len(args) == 0:
            return "Invalid argument: At least one <ID> is required.\n"

        for arg in args:
            if not arg.isdigit():
                return f"Invalid argument: {arg} is not a valid ID. It must be a valid integer to an existing task!\n"
            if not CliLlmModuleCore.is_valid_task(int(arg)):
                return f"Invalid argument: Task with ID {arg} does not exist.\n"
            task = CliLlmModuleCore.get_task(int(arg))
            if not task or not task.status == CliLlmTOTTaskStatus.PENDING:
                return f"Invalid argument: Task with ID {arg} is not pending.\n"

        result = "Marked the following tasks as failed:\n"
        for arg in args:
            task = CliLlmModuleCore.get_task(int(arg))
            assert task is not None
            task.status = CliLlmTOTTaskStatus.FAILED
            result += f"- Task {arg}: {task.title}\n"
        return result

    def desc(self) -> str:
        return (
            "Mark the specified tasks as failed. You can specify multiple IDs using ID1, ID2...\n"
            "USAGE: `/fail <ID1> <ID2> ... <IDN> [END]`\n"
            "EXAMPLE: `/fail 73 8 [END]`\n"
        )


#! Command: /time_of_creation ID
class CliLlmCommandTimeOfCreation(CliLlmCommandBase):
    def __init__(self):
        super().__init__("time_of_creation")

    def action(self, args: list[str]) -> str:
        if len(args) != 1:
            return "Invalid argument: <ID> is required.\n"

        task_id = args[0]

        if not task_id.isdigit():
            return f"Invalid argument: {task_id} is not a valid ID. It must be a valid integer to an existing task!\n"

        if not CliLlmModuleCore.is_valid_task(int(task_id)):
            return f"Invalid argument: Task with ID {task_id} does not exist.\n"

        task = CliLlmModuleCore.get_task(int(task_id))
        assert task is not None
        return f"Task {task_id} was created at {task.created_at}.\n"

    def desc(self) -> str:
        return (
            "Get the time of creation of the specified task.\n"
            "USAGE: `/time_of_creation <ID> [END]`\n"
            "EXAMPLE: `/time_of_creation 2 [END]`\n"
        )


#! Command: /time_of_update ID
class CliLlmCommandTimeOfUpdate(CliLlmCommandBase):
    def __init__(self):
        super().__init__("time_of_update")

    def action(self, args: list[str]) -> str:
        if len(args) != 1:
            return "Invalid argument: <ID> is required.\n"

        task_id = args[0]

        if not task_id.isdigit():
            return f"Invalid argument: {task_id} is not a valid ID. It must be a valid integer to an existing task!\n"

        if not CliLlmModuleCore.is_valid_task(int(task_id)):
            return f"Invalid argument: Task with ID {task_id} does not exist.\n"

        task = CliLlmModuleCore.get_task(int(task_id))
        assert task is not None
        return f"Task {task_id} was updated at {task.updated_at}.\n"

    def desc(self) -> str:
        return (
            "Get the time of update of the specified task.\n"
            "USAGE: `/time_of_update <ID> [END]`\n"
            "EXAMPLE: `/time_of_update 42 [END]`\n"
        )


#! Command: /pending_task_list
class CliLlmCommandPendingTaskList(CliLlmCommandBase):
    def __init__(self):
        super().__init__("pending_task_list")

    def action(self, args: list[str]) -> str:
        pending_tasks = CliLlmModuleCore.get_pending_tasks()
        if pending_tasks == []:
            return "No pending tasks.\n"

        result = "Pending tasks:\n"
        for task in pending_tasks:
            parents = ", ".join([str(p) for p in task.parents])
            childrens = ", ".join([str(c) for c in task.children])
            result += f"- ID: {task.id}, Title: {task.title}, Parents: {parents}, Childrens: {childrens}\n"
        return result

    def desc(self) -> str:
        return (
            "List all pending tasks with their IDs and titles.\n"
            "USAGE: `/pending_task_list [END]`\n"
            "EXAMPLE: `/pending_task_list [END]`\n"
        )


#! Command: /current_task
class CliLlmCommandCurrentTask(CliLlmCommandBase):
    def __init__(self):
        super().__init__("current_task")

    def action(self, args: list[str]) -> str:
        if CliLlmModuleCore.CURRENT_TASK == -1:
            return "No current task.\n"
        task = CliLlmModuleCore.get_task(CliLlmModuleCore.CURRENT_TASK)
        assert task is not None
        return f"Current task ID: {task.id}, Title: {task.title}\n"

    def desc(self) -> str:
        return (
            "Get the current task <ID> and title.\n"
            "USAGE: `/current_task [END]`\n"
            "EXAMPLE: `/current_task [END]`\n"
        )


#! Command: /childrens ID
class CliLlmCommandChildrens(CliLlmCommandBase):
    def __init__(self):
        super().__init__("childrens")

    def action(self, args: list[str]) -> str:
        if len(args) != 1:
            return "Invalid argument: <ID> is required.\n"

        task_id = args[0]

        if not task_id.isdigit():
            return f"Invalid argument: {task_id} is not a valid ID. It must be a valid integer to an existing task!\n"

        if not CliLlmModuleCore.is_valid_task(int(task_id)):
            return f"Invalid argument: Task with ID {task_id} does not exist.\n"

        task = CliLlmModuleCore.get_task(int(task_id))
        assert task is not None
        return f"Task {task_id} has childrens: {', '.join([str(c) for c in task.children])}\n"

    def desc(self) -> str:
        return (
            "Get the childrens of the specified task.\n"
            "USAGE: `/childrens <ID> [END]`\n"
            "EXAMPLE: `/childrens 68 [END]`\n"
        )


#! Command: /parents ID
class CliLlmCommandParents(CliLlmCommandBase):
    def __init__(self):
        super().__init__("parents")

    def action(self, args: list[str]) -> str:
        if len(args) != 1:
            return "Invalid argument: <ID> is required.\n"

        task_id = args[0]

        if not task_id.isdigit():
            return f"Invalid argument: {task_id} is not a valid ID. It must be a valid integer to an existing task!\n"

        if not CliLlmModuleCore.is_valid_task(int(task_id)):
            return f"Invalid argument: Task with ID {task_id} does not exist.\n"

        task = CliLlmModuleCore.get_task(int(task_id))
        assert task is not None
        return (
            f"Task {task_id} has parents: {', '.join([str(p) for p in task.parents])}\n"
        )

    def desc(self) -> str:
        return (
            "Get the parents of the specified task.\n"
            "USAGE: `/parents <ID> [END]`\n"
            "EXAMPLE: `/parents 56 [END]`\n"
        )


#! Command: /status ID
class CliLlmCommandStatus(CliLlmCommandBase):
    def __init__(self):
        super().__init__("status")

    def action(self, args: list[str]) -> str:
        if len(args) != 1:
            return "Invalid argument: <ID> is required.\n"

        task_id = args[0]

        if not task_id.isdigit():
            return f"Invalid argument: {task_id} is not a valid ID. It must be a valid integer to an existing task!\n"

        if not CliLlmModuleCore.is_valid_task(int(task_id)):
            return f"Invalid argument: Task with ID {task_id} does not exist.\n"

        task = CliLlmModuleCore.get_task(int(task_id))
        assert task is not None
        return f"Task {task_id} status: {CliLlmTOTTaskStatus.to_string(task.status)}\n"

    def desc(self) -> str:
        return (
            "Get the status of the specified task.\n"
            "USAGE: `/status <ID> [END]`\n"
            "EXAMPLE: `/status 53 [END]`\n"
        )


#! Command: /history <ID> [END]
class CliLlmCommandHistory(CliLlmCommandBase):
    def __init__(self):
        super().__init__("history")

    def action(self, args: list[str]) -> str:
        if len(args) != 1:
            return "Invalid argument: <ID> is required.\n"

        task_id = args[0]

        if not task_id.isdigit():
            return f"Invalid argument: {task_id} is not a valid ID. It must be a valid integer to an existing task!\n"

        if not CliLlmModuleCore.is_valid_task(int(task_id)):
            return f"Invalid argument: Task with <ID> {task_id} does not exist.\n"

        task = CliLlmModuleCore.get_task(int(task_id))
        assert task is not None
        return f"Task {task_id} history:\n" + "\n".join(task.history) + "\n"

    def desc(self) -> str:
        return (
            "Get the comment / reason history of the specified task.\n"
            "USAGE: `/history <ID> [END]`\n"
            "EXAMPLE: `/history 39 [END]`\n"
        )


#! Command: /ask <MESSAGE> [END]
class CliLlmCommandAsk(CliLlmCommandBase):
    def __init__(self):
        super().__init__("ask")

    def action(self, args: list[str]) -> str:
        if not args:
            return "Invalid argument: <MESSAGE> is required."
        message = " ".join(args[:-1]).strip()
        if not message:
            return "Invalid argument: Empty message."
        return f"Question asked: {message}"

    def desc(self) -> str:
        return (
            "Ask a question to the user and wait for an answer. The AI will pause until the user responds.\n"
            "USAGE: `/ask <MESSAGE> [END]`\n"
            "EXAMPLE: `/ask What is the capital of France? [END]`\n"
        )


#!Module Core -> Task Status
class CliLlmTOTTaskStatus(enumerate):
    PENDING = 0
    SUCCESS = 1
    FAILED = 2

    @staticmethod
    def to_string(status: int):
        match status:
            case CliLlmTOTTaskStatus.PENDING:
                return "PENDING"
            case CliLlmTOTTaskStatus.SUCCESS:
                return "SUCCESS"
            case CliLlmTOTTaskStatus.FAILED:
                return "FAILED"
            case _:
                return "UNKNOWN"


#!Module Core -> Task
class CliLlmTOTTask:
    def __init__(self, task_id: int, title: str):
        self.id = task_id
        self.title = title
        self.parents = []
        self.children = []
        self.history: list[str] = []
        self.status = CliLlmTOTTaskStatus.PENDING
        self.created_at = datetime.datetime.now()
        self.updated_at = datetime.datetime.now()


#!Module Core
class CliLlmModuleCore(CliLlmModuleBase):
    TASKS: dict[int, CliLlmTOTTask] = {}
    CURRENT_TASK: int = -1

    def __init__(self):
        super().__init__("core")

    @staticmethod
    def create_task(title: str) -> int:
        task_id = len(CliLlmModuleCore.TASKS)
        task = CliLlmTOTTask(task_id, title)
        CliLlmModuleCore.TASKS[task_id] = task
        CliLlmModuleCore.CURRENT_TASK = task_id
        return task_id

    @staticmethod
    def has_created_any_task() -> bool:
        return len(CliLlmModuleCore.TASKS) > 0

    @staticmethod
    def has_done_all_tasks() -> bool:
        for task in CliLlmModuleCore.TASKS.values():
            if task.status == CliLlmTOTTaskStatus.PENDING:
                return False
        return True

    @staticmethod
    def is_valid_task(task_id: int) -> bool:
        return task_id in CliLlmModuleCore.TASKS

    @staticmethod
    def add_dependency(children: int, parent: int):
        assert CliLlmModuleCore.is_valid_task(parent)
        assert CliLlmModuleCore.is_valid_task(children)
        CliLlmModuleCore.TASKS[parent].children.append(children)
        CliLlmModuleCore.TASKS[children].parents.append(parent)

    @staticmethod
    def get_task(task_id: int) -> CliLlmTOTTask | None:
        return (
            CliLlmModuleCore.TASKS[task_id]
            if task_id in CliLlmModuleCore.TASKS
            else None
        )

    @staticmethod
    def get_pending_tasks() -> list[CliLlmTOTTask]:
        pending_list = []
        for task in CliLlmModuleCore.TASKS.values():
            if task.status == CliLlmTOTTaskStatus.PENDING:
                pending_list.append(task)
        return pending_list

    @staticmethod
    def set_current_task(task_id: int):
        assert CliLlmModuleCore.is_valid_task(task_id)
        CliLlmModuleCore.CURRENT_TASK = task_id

    def retrieve_commands(self) -> list[CliLlmCommandBase]:
        return [
            CliLlmCommandExit(),
            CliLlmCommandReason(),
            CliLlmCommandCreateTask(),
            CliLlmCommandComment(),
            CliLlmCommandDone(),
            CliLlmCommandFail(),
            CliLlmCommandTimeOfCreation(),
            CliLlmCommandTimeOfUpdate(),
            CliLlmCommandPendingTaskList(),
            CliLlmCommandCurrentTask(),
            CliLlmCommandChildrens(),
            CliLlmCommandParents(),
            CliLlmCommandStatus(),
            CliLlmCommandHistory(),
            CliLlmCommandAsk(),
        ]


#!Command /open_file <FILE> [END]
class CliLlmCommandOpenFile(CliLlmCommandBase):
    def __init__(self):
        super().__init__("open_file")

    def action(self, args: list[str]) -> str:
        if not args:
            return "Error: <PATH> is required."
        path = args[0]
        full_path = os.path.abspath(os.path.join(CliLlmKernel.CURRENT_WD, path))
        # Validate path is allowed
        allowed = False
        for dir in CliLlmKernel.ALLOWED_DIRECTORIES:
            if full_path.startswith(dir + os.sep) or full_path == dir:
                allowed = True
                break
        if not allowed:
            return f"Error: Path '{path}' is not allowed."
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        try:
            if CliLlmModuleIO.CURRENT_FILE:
                CliLlmModuleIO.CURRENT_FILE.close()
            CliLlmModuleIO.CURRENT_FILE = open(full_path, "r+")
            CliLlmModuleIO.CURRENT_FILE_PATH = full_path
            return f"File opened: {full_path}"
        except Exception as e:
            return f"Error opening file: {str(e)}"

    def desc(self) -> str:
        return (
            "Open a file for reading/writing. Only one file can be open at a time.\n"
            "This will create the parent directories automatically if missing.\n"
            "USAGE: `/open_file <PATH> [END]\n"
            "EXAMPLE: `/open_file report.txt [END]\n"
        )


#!Command /close_file [END]
class CliLlmCommandCloseFile(CliLlmCommandBase):
    def __init__(self):
        super().__init__("close_file")

    def action(self, args: list[str]) -> str:
        if not CliLlmModuleIO.CURRENT_FILE:
            return "No file is currently open."
        try:
            CliLlmModuleIO.CURRENT_FILE.close()
            CliLlmModuleIO.CURRENT_FILE = None
            CliLlmModuleIO.CURRENT_FILE_PATH = None
            return "File closed successfully."
        except Exception as e:
            return f"Error closing file: {str(e)}"

    def desc(self) -> str:
        return (
            "Close the currently open file.\n"
            "USAGE: `/close_file [END]\n"
            "EXAMPLE: `/close_file [END]\n"
        )


#!Command /cd <DIR> [END]
class CliLlmCommandCD(CliLlmCommandBase):
    def __init__(self):
        super().__init__("cd")

    def action(self, args: list[str]) -> str:
        if not args:
            return "Error: <PATH> is required."
        path = args[0]
        target = CliLlmModuleIO.compute_absolute_path(path)
        allowed = False
        for dir in CliLlmKernel.ALLOWED_DIRECTORIES:
            if target.startswith(dir + os.sep) or target == dir:
                allowed = True
                break
        if not allowed:
            absolute_paths = [
                os.path.abspath(d) for d in CliLlmKernel.ALLOWED_DIRECTORIES
            ]
            return f"Error: Path '{path}' is not allowed. Allowed paths base are: {absolute_paths}"

        try:
            os.chdir(target)
            CliLlmKernel.CURRENT_WD = target
            return f"Changed directory to: {target}"
        except Exception as e:
            return f"Error changing directory: {str(e)}"

    def desc(self) -> str:
        return (
            "Change the current working directory.\n"
            "USAGE: `/cd <PATH> [END]\n"
            "EXAMPLE: `/cd project/ [END]\n"
        )


#!Command /move <SOURCE> <DESTINATION> [END]
class CliLlmCommandMove(CliLlmCommandBase):
    def __init__(self):
        super().__init__("move")

    def action(self, args: list[str]) -> str:
        if len(args) < 2:
            return "Error: Need <SOURCE> and <DESTINATION>."
        src = args[0]
        dst = args[1]
        src_abs = CliLlmModuleIO.compute_absolute_path(src)
        dst_abs = CliLlmModuleIO.compute_absolute_path(dst)
        # Validate both paths are allowed
        allowed_src = any(
            [
                CliLlmModuleIO.is_allowed_path(d)
                for d in CliLlmKernel.ALLOWED_DIRECTORIES
            ]
        )
        allowed_dst = any(
            [
                CliLlmModuleIO.is_allowed_path(d)
                for d in CliLlmKernel.ALLOWED_DIRECTORIES
            ]
        )
        if not allowed_src or not allowed_dst:
            return "Error: Source or destination is not allowed."
        try:
            shutil.move(src_abs, dst_abs)
            return f"Moved '{src_abs}' to '{dst_abs}'"
        except Exception as e:
            return f"Error moving: {str(e)}"

    def desc(self) -> str:
        return (
            "Move a file or directory.\n"
            "USAGE: `/move <SOURCE> <DESTINATION> [END]\n"
            "EXAMPLE: `/move file.txt backup/ [END]\n"
        )


#!Command /cursor [END]
class CliLlmCommandCursor(CliLlmCommandBase):
    def __init__(self):
        super().__init__("cursor")

    def action(self, args: list[str]) -> str:
        if not CliLlmModuleIO.CURRENT_FILE:
            return "No file is open. Use /open_file first."
        try:
            pos = CliLlmModuleIO.CURRENT_FILE.tell()
            return f"Current cursor position: {pos} bytes"
        except Exception as e:
            return f"Error: {str(e)}"

    def desc(self) -> str:
        return (
            "Get the current file cursor position.\n"
            "USAGE: `/cursor [END]\n"
            "EXAMPLE: `/cursor [END]\n"
        )


#!Command /move_cursor <POSITION> [END]
class CliLlmCommandMoveCursor(CliLlmCommandBase):
    def __init__(self):
        super().__init__("move_cursor")

    def action(self, args: list[str]) -> str:
        if not args:
            return "Error: <POSITION> is required."
        try:
            pos = int(args[0])
        except Exception as _:
            return "Error: Position must be an integer."
        if not CliLlmModuleIO.CURRENT_FILE:
            return "No file is open. Use /open_file first."
        try:
            CliLlmModuleIO.CURRENT_FILE.seek(pos)
            return f"Cursor moved to {pos}"
        except Exception as e:
            return f"Error: {str(e)}"

    def desc(self) -> str:
        return (
            "Move the file cursor to a specific position.\n"
            "USAGE: `/move_cursor <POSITION> [END]\n"
            "EXAMPLE: `/move_cursor 1024 [END]\n"
        )


#!Command /max_size <SIZE> [END]
class CliLlmCommandMaxSize(CliLlmCommandBase):
    def __init__(self):
        super().__init__("max_size")

    def action(self, args: list[str]) -> str:
        if not CliLlmModuleIO.CURRENT_FILE:
            return "No file is open. Use /open_file first."
        try:
            CliLlmModuleIO.CURRENT_FILE.seek(0, os.SEEK_END)
            size = CliLlmModuleIO.CURRENT_FILE.tell()
            return f"File size: {size} bytes"
        except Exception as e:
            return f"Error: {str(e)}"

    def desc(self) -> str:
        return (
            "Get the current file's maximum size.\n"
            "USAGE: `/max_size [END]\n"
            "EXAMPLE: `/max_size [END]\n"
        )


#!Command /read <SIZE> [END]
class CliLlmCommandRead(CliLlmCommandBase):
    def __init__(self):
        super().__init__("read")

    def action(self, args: list[str]) -> str:
        if len(args) < 2:
            return "Error: Need <START> and <END> positions."
        try:
            start = int(args[0])
            end = int(args[1])
        except Exception as _:
            return "Error: Positions must be integers."
        if not CliLlmModuleIO.CURRENT_FILE:
            return "No file is open. Use /open_file first."
        try:
            CliLlmModuleIO.CURRENT_FILE.seek(start)
            data = CliLlmModuleIO.CURRENT_FILE.read(end - start)
            if len(data) > 4096:
                return "Error: Exceeds 4096 character limit."
            return f"Read data: {data}"
        except Exception as e:
            return f"Error: {str(e)}"

    def desc(self) -> str:
        return (
            "Read up to 4096 characters from a file.\n"
            "USAGE: `/read <START> <END> [END]\n"
            "EXAMPLE: `/read 0 4095 [END]\n"
        )


#!Command /write <DATA> [END]
class CliLlmCommandWrite(CliLlmCommandBase):
    def __init__(self):
        super().__init__("write")

    def action(self, args: list[str]) -> str:
        if not args:
            return "Error: <CONTENT> is required."
        content = " ".join(args)
        if not CliLlmModuleIO.CURRENT_FILE:
            return "No file is open. Use /open_file first."
        try:
            current_pos = CliLlmModuleIO.CURRENT_FILE.tell()
            CliLlmModuleIO.CURRENT_FILE.write(content)
            return f"Wrote {len(content)} bytes at {current_pos}"
        except Exception as e:
            return f"Error: {str(e)}"

    def desc(self) -> str:
        return (
            "Write content to the current file position.\n"
            "USAGE: `/write <CONTENT> [END]\n"
            "EXAMPLE: `/write 'Hello World' [END]\n"
        )


#!Command /grep <REGEX> [END]
class CliLlmCommandGrep(CliLlmCommandBase):
    def __init__(self):
        super().__init__("grep")

    def action(self, args: list[str]) -> str:
        if not args:
            return "Error: <REGEX> is required."
        pattern = " ".join(args)
        if not CliLlmModuleIO.CURRENT_FILE:
            return "No file is open. Use /open_file first."
        try:
            CliLlmModuleIO.CURRENT_FILE.seek(0)
            content = CliLlmModuleIO.CURRENT_FILE.read(4096)
            matches = re.findall(pattern, content)
            return f"Matches: {matches}" if matches else "No matches found."
        except Exception as e:
            return f"Error: {str(e)}"

    def desc(self) -> str:
        return (
            "Search for regex patterns in the current file.\n"
            "USAGE: `/grep <REGEX_PATTERN> [END]\n"
            "EXAMPLE: `/grep 'error' [END]\n"
        )


#!Command /replace <OLD> <NEW> [END]
class CliLlmCommandReplace(CliLlmCommandBase):
    def __init__(self):
        super().__init__("replace")

    def action(self, args: list[str]) -> str:
        if len(args) < 3:
            return "Error: Need <START>, <END>, and <CONTENT>."
        try:
            start = int(args[0])
            end = int(args[1])
            new_content = " ".join(args[2:])
        except Exception as _:
            return "Error: Invalid parameters."
        if not CliLlmModuleIO.CURRENT_FILE:
            return "No file is open. Use /open_file first."
        try:
            CliLlmModuleIO.CURRENT_FILE.seek(0)
            data = CliLlmModuleIO.CURRENT_FILE.read()
            new_data = data[:start] + new_content + data[end:]
            CliLlmModuleIO.CURRENT_FILE.seek(0)
            CliLlmModuleIO.CURRENT_FILE.write(new_data)
            CliLlmModuleIO.CURRENT_FILE.truncate()
            return "Content replaced successfully."
        except Exception as e:
            return f"Error: {str(e)}"

    def desc(self) -> str:
        return (
            "Replace content between positions.\n"
            "USAGE: `/replace <START> <END> <NEW_CONTENT> [END]\n"
            "EXAMPLE: `/replace 10 20 'new text' [END]\n"
        )


#!Command /replace_regex <REGEX> <NEW> [END]
class CliLlmCommandReplaceRegex(CliLlmCommandBase):
    def __init__(self):
        super().__init__("replace_regex")

    def action(self, args: list[str]) -> str:
        if len(args) < 2:
            return "Error: Need <PATTERN> and <SUBSTITUTION>."
        pattern = args[0]
        sub = " ".join(args[1:])
        if not CliLlmModuleIO.CURRENT_FILE:
            return "No file is open. Use /open_file first."
        try:
            CliLlmModuleIO.CURRENT_FILE.seek(0)
            data = CliLlmModuleIO.CURRENT_FILE.read()
            new_data = re.sub(pattern, sub, data)
            CliLlmModuleIO.CURRENT_FILE.seek(0)
            CliLlmModuleIO.CURRENT_FILE.write(new_data)
            CliLlmModuleIO.CURRENT_FILE.truncate()
            return "Regex substitution applied."
        except Exception as e:
            return f"Error: {str(e)}"

    def desc(self) -> str:
        return (
            "Replace content using regex.\n"
            "USAGE: `/replace_regex <PATTERN> <SUBSTITUTION> [END]\n"
            "EXAMPLE: `/replace_regex 'old' 'new' [END]\n"
        )


#!Command /replace_regex_range <START> <END> <NEW> [END]
class CliLlmCommandReplaceRegexRange(CliLlmCommandBase):
    def __init__(self):
        super().__init__("replace_regex_range")

    def action(self, args: list[str]) -> str:
        if len(args) < 4:
            return "Error: Need <START>, <END>, <PATTERN>, <SUBSTITUTION>."
        try:
            start = int(args[0])
            end = int(args[1])
            pattern = args[2]
            sub = args[3]
        except Exception as _:
            return "Error: Invalid parameters."
        if not CliLlmModuleIO.CURRENT_FILE:
            return "No file is open. Use /open_file first."
        try:
            CliLlmModuleIO.CURRENT_FILE.seek(0)
            data = CliLlmModuleIO.CURRENT_FILE.read()
            target = data[start:end]
            new_target = re.sub(pattern, sub, target)
            new_data = data[:start] + new_target + data[end:]
            CliLlmModuleIO.CURRENT_FILE.seek(0)
            CliLlmModuleIO.CURRENT_FILE.write(new_data)
            CliLlmModuleIO.CURRENT_FILE.truncate()
            return "Regex substitution applied in range."
        except Exception as e:
            return f"Error: {str(e)}"

    def desc(self) -> str:
        return (
            "Apply regex substitution within a range.\n"
            "USAGE: `/replace_regex_range <START> <END> <PATTERN> <SUBSTITUTION> [END]\n"
            "EXAMPLE: `/replace_regex_range 0 100 'old' 'new' [END]\n"
        )


#!Command /erase <START> <END> [END]
class CliLlmCommandErase(CliLlmCommandBase):
    def __init__(self):
        super().__init__("erase")

    def action(self, args: list[str]) -> str:
        if not args or not args[0].isdigit():
            return "Error: <BYTES> is required (integer)."
        bytes_to_erase = int(args[0])
        if not CliLlmModuleIO.CURRENT_FILE:
            return "No file is open. Use /open_file first."
        try:
            current_pos = CliLlmModuleIO.CURRENT_FILE.tell()
            CliLlmModuleIO.CURRENT_FILE.seek(0)
            data = CliLlmModuleIO.CURRENT_FILE.read()
            new_data = data[:current_pos] + data[current_pos + bytes_to_erase :]
            CliLlmModuleIO.CURRENT_FILE.seek(0)
            CliLlmModuleIO.CURRENT_FILE.write(new_data)
            CliLlmModuleIO.CURRENT_FILE.truncate()
            CliLlmModuleIO.CURRENT_FILE.seek(current_pos)  # Reset cursor
            return f"Erased {bytes_to_erase} bytes at position {current_pos}"
        except Exception as e:
            return f"Error: {str(e)}"

    def desc(self) -> str:
        return (
            "Erase bytes from the current file's cursor position.\n"
            "USAGE: `/erase <BYTES> [END]\n"
            "EXAMPLE: `/erase 10 [END]\n"
        )


#!Command /list_dir <DIR> [END]
class CliLlmCommandListDir(CliLlmCommandBase):
    def __init__(self):
        super().__init__("list_dir")

    def action(self, args: list[str]) -> str:
        path = args[0] if args else "."
        full_path = os.path.abspath(os.path.join(CliLlmKernel.CURRENT_WD, path))
        if not CliLlmModuleIO.is_allowed_path(full_path):
            return f"Error: Path '{path}' is not allowed."
        try:
            entries = os.listdir(full_path)
            return f"Contents of '{full_path}':\n- " + "\n- ".join(entries)
        except Exception as e:
            return f"Error: {str(e)}"

    def desc(self) -> str:
        return (
            "List directory contents.\n"
            "USAGE: `/list_dir [PATH] [END]\n"
            "EXAMPLE: `/list_dir docs/ [END]\n"
        )


#!Command /remove <FILE> [END]
class CliLlmCommandRemove(CliLlmCommandBase):
    def __init__(self):
        super().__init__("remove")

    def action(self, args: list[str]) -> str:
        if not args:
            return "Error: <PATH> is required."
        target = args[0]
        full_path = os.path.abspath(os.path.join(CliLlmKernel.CURRENT_WD, target))
        if not CliLlmModuleIO.is_allowed_path(full_path):
            return f"Error: Path '{target}' is not allowed."
        try:
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
            else:
                os.remove(full_path)
            return f"Removed: {full_path}"
        except Exception as e:
            return f"Error: {str(e)}"

    def desc(self) -> str:
        return (
            "Delete a file or directory.\n"
            "USAGE: `/remove <PATH> [END]\n"
            "EXAMPLE: `/remove temp.txt [END]\n"
        )


#!Command /cat <FILE> [END]
class CliLlmCommandCat(CliLlmCommandBase):
    def __init__(self):
        super().__init__("cat")

    def action(self, args: list[str]) -> str:
        if not args:
            return "Error: <PATH> is required."
        path = args[0]
        full_path = os.path.abspath(os.path.join(CliLlmKernel.CURRENT_WD, path))
        if not CliLlmModuleIO.is_allowed_path(full_path):
            return f"Error: Path '{path}' is not allowed."
        try:
            with open(full_path, "r") as f:
                content = f.read(4096)  # Limit to 4KB to prevent overflow
                return f"File contents of '{full_path}':\n{content}"
        except Exception as e:
            return f"Error: {str(e)}"

    def desc(self) -> str:
        return (
            "Display file contents (max 4096 characters).\n"
            "USAGE: `/cat <PATH> [END]\n"
            "EXAMPLE: `/cat config.txt [END]\n"
        )


#!Command /file_info <FILE> [END]
class CliLlmCommandFileInfo(CliLlmCommandBase):
    def __init__(self):
        super().__init__("file_info")

    def action(self, args: list[str]) -> str:
        if not CliLlmModuleIO.CURRENT_FILE_PATH:
            return "No file is open. Use /open_file first."
        try:
            stats = os.stat(CliLlmModuleIO.CURRENT_FILE_PATH)
            return (
                f"File: {CliLlmModuleIO.CURRENT_FILE_PATH}\n"
                f"Size: {stats.st_size} bytes\n"
                f"Created: {datetime.datetime.fromtimestamp(stats.st_birthtime)}\n"
                f"Modified: {datetime.datetime.fromtimestamp(stats.st_mtime)}"
            )
        except Exception as e:
            return f"Error: {str(e)}"

    def desc(self) -> str:
        return (
            "Show metadata for the currently open file.\n"
            "USAGE: `/file_info [END]\n"
            "EXAMPLE: `/file_info [END]\n"
        )


#!Command /truncate <SIZE> [END]
class CliLlmCommandTruncate(CliLlmCommandBase):
    def __init__(self):
        super().__init__("truncate")

    def action(self, args: list[str]) -> str:
        if not args or not args[0].isdigit():
            return "Error: <SIZE> is required (integer)."
        size = int(args[0])
        if not CliLlmModuleIO.CURRENT_FILE:
            return "No file is open. Use /open_file first."
        try:
            CliLlmModuleIO.CURRENT_FILE.truncate(size)
            return f"File truncated to {size} bytes"
        except Exception as e:
            return f"Error: {str(e)}"

    def desc(self) -> str:
        return (
            "Resize the current file to a specific size.\n"
            "USAGE: `/truncate <SIZE> [END]\n"
            "EXAMPLE: `/truncate 1024 [END]\n"
        )


#!Command /copy <SOURCE> <DESTINATION> [END]
class CliLlmCommandCopy(CliLlmCommandBase):
    def __init__(self):
        super().__init__("copy")

    def action(self, args: list[str]) -> str:
        if len(args) < 2:
            return "Error: Need <SOURCE> and <DESTINATION>."
        src = args[0]
        dst = args[1]
        src_abs = CliLlmModuleIO.compute_absolute_path(src)
        dst_abs = CliLlmModuleIO.compute_absolute_path(dst)
        if not CliLlmModuleIO.is_allowed_path(
            src_abs
        ) or not CliLlmModuleIO.is_allowed_path(dst_abs):
            return "Error: Path is not allowed."
        try:
            if os.path.isdir(src_abs):
                shutil.copytree(src_abs, dst_abs)
            else:
                shutil.copy2(src_abs, dst_abs)
            return f"Copied '{src_abs}' to '{dst_abs}'"
        except Exception as e:
            return f"Error: {str(e)}"

    def desc(self) -> str:
        return (
            "Copy a file or directory.\n"
            "USAGE: `/copy <SOURCE> <DESTINATION> [END]\n"
            "EXAMPLE: `/copy config.txt config.bak [END]\n"
        )


#!Module IO
class CliLlmModuleIO(CliLlmModuleBase):
    CURRENT_FILE: TextIO | None = None
    CURRENT_FILE_PATH: str | None = None
    ALLOWED_DIRECTORIES: list[str] = []

    def __init__(self):
        super().__init__("io", dependencies=["core"])  # Depends on core module

    def retrieve_commands(self) -> list[CliLlmCommandBase]:
        return [
            CliLlmCommandOpenFile(),
            CliLlmCommandCloseFile(),
            CliLlmCommandCD(),
            CliLlmCommandMove(),
            CliLlmCommandCursor(),
            CliLlmCommandMoveCursor(),
            CliLlmCommandMaxSize(),
            CliLlmCommandRead(),
            CliLlmCommandWrite(),
            CliLlmCommandGrep(),
            CliLlmCommandReplace(),
            CliLlmCommandReplaceRegex(),
            CliLlmCommandReplaceRegexRange(),
            CliLlmCommandErase(),
            CliLlmCommandListDir(),
            CliLlmCommandRemove(),
            CliLlmCommandCat(),
            CliLlmCommandFileInfo(),
            CliLlmCommandTruncate(),
            CliLlmCommandCopy(),
        ]

    @staticmethod
    def compute_absolute_path(path: str):
        if path.startswith("."):
            return os.path.abspath(os.path.join(CliLlmKernel.CURRENT_WD, path))
        else:
            return os.path.abspath(path)

    @staticmethod
    def is_allowed_path(path: str) -> bool:
        abs_path = CliLlmModuleIO.compute_absolute_path(path)
        for dir in CliLlmKernel.ALLOWED_DIRECTORIES:
            if abs_path.startswith(dir + os.sep) or abs_path == dir:
                return True
        return False


# TODO: Module Coding Tools:
# This module allows the AI to use LSP tools such as treesitter, but also grep

# TODO: Coding Tools
# /symbols FILE
# /workspace_symbols FILE
# /grep_symbols FILE REGEX
# /workspace_grep_symbols FILE REGEX
# /replace_symbols SYMBOL NEW_SYMBOL

# TODO: Module Execute:
# This module allows the AI to execute scripts, executable within the allowed folders

# TODO: Execute commands:
# /execute PATH ARGS

# TODO: Module Web:
# This module allows the AI to use a web browser, search for information on the web, etc...
# /web_search QUERY

# TODO: Module Scratchpad:
# This module allows the AI to use a scratchpad, a temporary space where it can write code, notes, etc...
# /memorize MESSAGE
# /task_memorize TASK_ID MESSAGE
# /remember TASK_ID
# /memory_list
# /memory_clear

# TODO: Module Debug:
# This module allows the AI to use a debugger on the currently opened file, set breakpoints, etc... It uses the IO module to get the current file.
# /debug PATH DARGS ARGS (automatically choose gdb, pydbg etc...)


###! Kernel
class CliLlmKernel:
    SHOULD_EXIT: bool = False
    CONTEXT_HISTORY = CliLlmMessageHistory()
    COMMANDS = CliLlmCommandManager()
    MODULES = CliLlmModuleManager()
    MODEL = None
    CURRENT_WD = os.getcwd()
    TASK: str = "There is no task to do."

    @staticmethod
    def parse_args(argv: list[str]):
        parser = argparse.ArgumentParser(
            description="Launch the Pulsar Project kernel."
        )
        parser.add_argument(
            "--modules",
            nargs="*",
            help="List of module names to load (e.g., --modules core io)",
        )
        parser.add_argument(
            "--initial-task",
            type=str,
            help="Specify the initial task to execute after kernel initialization.",
            required=True,
        )
        parser.add_argument(
            "--allowed-dirs",
            type=str,
            help="Comma-separated list of allowed directories for the IO module.",
            default=".",
        )
        parsed_args = parser.parse_args(argv)

        # Register modules
        module_names = parsed_args.modules or []
        for name in module_names:
            match name:
                case "core":
                    CliLlmKernel.MODULES.register(CliLlmModuleCore())

                case "io":
                    CliLlmKernel.MODULES.register(CliLlmModuleIO())

                case _:
                    raise ValueError(f"Module '{name}' is not recognized.")

        # Process allowed directories
        if parsed_args.allowed_dirs:
            CliLlmKernel.ALLOWED_DIRECTORIES = [
                os.path.abspath(d) for d in parsed_args.allowed_dirs.split(",")
            ]
        else:
            CliLlmKernel.ALLOWED_DIRECTORIES = [os.getcwd()]

            # Specify initial task
            CliLlmKernel.TASK = parsed_args.initial_task

    @staticmethod
    def get_wd() -> str:
        return CliLlmKernel.CURRENT_WD

    @staticmethod
    def init():
        CliLlmModel.init()
        print("\nPulsar Project - CLI LLM\n")

        # Initialize core module by default
        if not CliLlmKernel.MODULES.is_registered("core"):
            CliLlmKernel.MODULES.register(CliLlmModuleCore())

    @staticmethod
    def clean():
        CliLlmKernel.SHOULD_EXIT = True
        CliLlmKernel.CONTEXT_HISTORY = None
        CliLlmKernel.COMMANDS.clear()
        CliLlmModel.finish()

    @staticmethod
    def get_command_list_str():
        commands_str = ""
        for cmd in CliLlmKernel.COMMANDS.get_command_list():
            commands_str += f"- /{cmd.name}:\n{cmd.desc()}"
        return commands_str.strip()

    @staticmethod
    def compute_sys_prompt():
        return (
            "[INST] <s>\n"
            "**SYSTEM INSTRUCTIONS**\n\n"
            "You are a structured reasoning agent. **ALL ACTIONS MUST BE PERFORMED THROUGH THE EXPLICITLY LISTED COMMANDS ONLY.**\n\n"
            "## PROTOCOL ENFORCEMENT\n"
            "1. **COMMAND RESTRICTION**: Only use the commands listed under AVAILABLE COMMANDS. Direct task-solving without commands is prohibited.\n"
            "2. **COMMAND TERMINATION**: Every command must end with `[END]`. Failure to do so will result in rejection.\n"
            "3. **SEQUENTIAL FLOW**: Follow the protocol steps below *for every action* without skipping steps.\n\n"
            "## STRUCTURED REASONING PROTOCOL\n"
            "1. **THINK** (REQUIRED):\n"
            "   - Use `/say` to document all reasoning, hypotheses, and plans.\n"
            "   - Analyze dependencies using `/parents` and `/childrens` before finalizing actions.\n"
            "   - **MUST** call `/say` at least once before any `/create_task` or `/done`.\n\n"
            "2. **ACT** (REQUIRED):\n"
            "   - Execute *exactly one* command from **Available Commands**.\n"
            "   - Example: `/create_task -c 5 -t Verify solution [END]`\n\n"
            "3. **REFLECT** (REQUIRED):\n"
            "   - Validate command results using status checks (e.g., `/status 5`, `/history 5`).\n"
            "   - If unsuccessful, use `/fail` and return to THINK phase.\n\n"
            "4. **PLAN** (REQUIRED):\n"
            "   - Adjust strategy using `/comment` to document reflections.\n"
            "   - Ensure all tasks meet dependency requirements before marking as done.\n\n"
            "## CURRENT TASK CONTEXT\n"
            f"**Main Objective**: {CliLlmKernel.TASK}\n\n"
            "## PROTOCOL EXAMPLES\n"
            "### Example 1: Mathematical Problem\n"
            "/say Analyzing equation: Isolate x by dividing both sides. Testing with substitution... [END]\n"
            "/create_task -c 3 -t Verify solution with x=5 [END]\n"
            "/comment 4 Substitution shows inconsistency at step 2 [END]\n\n"
            "### Example 2: Programming Task\n"
            "/say Infinite loop likely from off-by-one error in loop condition [END]\n"
            "/create_task -c 7 -t Implement safeguard counter [END]\n"
            "/done 7 [END]\n\n"
            "## STRICT RULES\n"
            "- **NO DIRECT TASK-SOLVING**: All actions must go through commands.\n"
            "- **MAX 3 SUBTASKS**: Do not create more than 3 subtasks per action.\n"
            "- **DEPENDENCY CHECK**: Use `/parents` before marking tasks as done.\n"
            "- **COMMENT REQUIREMENT**: Every `/create_task` must be followed by `/comment`.\n\n"
            "## AVAILABLE COMMANDS\n"
            f"{CliLlmKernel.get_command_list_str()}\n\n"
            "**VIOLATIONS OF THIS PROTOCOL WILL RESULT IN SYSTEM REJECTION.**\n"
            "</s> [/INST]"
        )

    @staticmethod
    def compute_continuous_history():
        assert CliLlmKernel.CONTEXT_HISTORY is not None
        return f"{CliLlmKernel.CONTEXT_HISTORY.get()}\n"

    @staticmethod
    def compute_next_response(prompt: str) -> Tuple[str, str]:
        # Generate first answer
        full_response = CliLlmModel.send_prompt(prompt)
        argv = full_response["choices"][0]["text"]
        argv = argv.strip()
        answer = f"\n{CliLlmKernel.get_wd()}$ /{argv} [END]\n"
        print(answer)
        print("")

        # Process first command
        argv_words = argv.split(" ")
        command = argv_words[0]
        result = CliLlmKernel.COMMANDS.execute(argv_words[0], argv_words[1:])

        # If command is `ask`, wait for user input
        if command == "ask":
            if len(argv_words) < 2 or argv_words[-1] != "[END]":
                print("Invalid ask command format.")
            else:
                question = " ".join(argv_words[1:-1])
                user_answer: str | None = None
                while user_answer is None:
                    user_answer = input(f"\nAI's question: {question}\nYour answer: ")
                CliLlmKernel.CONTEXT_HISTORY.add(f"User: {user_answer}")

        return result, answer

    @staticmethod
    def init_run():
        # System Prompt
        sys_prompt = CliLlmKernel.compute_sys_prompt()
        print(sys_prompt)
        print("")

        # Generate first answer
        result, answer = CliLlmKernel.compute_next_response(sys_prompt)

        # Initialize History
        CliLlmKernel.CONTEXT_HISTORY.set_system_prompt(sys_prompt)
        CliLlmKernel.CONTEXT_HISTORY.add(answer + "[INST]" + result + "[/INST]\n")

    @staticmethod
    def run():
        assert CliLlmKernel.CONTEXT_HISTORY is not None

        # Continuous Prompt
        prompt = CliLlmKernel.compute_continuous_history()

        # Appending working directory to generated prompt
        prompt += f"\n{CliLlmKernel.get_wd()}$ /"

        # Generate answer
        result, answer = CliLlmKernel.compute_next_response(prompt)

        # Update history
        CliLlmKernel.CONTEXT_HISTORY.add(answer + "[INST]" + result + "[/INST]\n")

    @staticmethod
    def loop():
        CliLlmKernel.init_run()
        while not CliLlmKernel.SHOULD_EXIT:
            CliLlmKernel.run()


###! Main
if __name__ == "__main__":
    # TODO: Maybe split the code into two CLIs, one for the human where he can:
    # - Change the task
    # - Finish the execution
    # - Pause the execution
    # - Answer AI question of the TOT commands module ($ cmd.ask command for TOT

    CliLlmKernel.parse_args(sys.argv[1:])
    CliLlmKernel.init()
    CliLlmKernel.loop()
