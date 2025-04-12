from llama_cpp import LlamaGrammar, Llama

import datetime
import os
import sys
import argparse
import json


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
    def send_prompt(prompt: str, grammar: LlamaGrammar | None, max_tokens: int = N_CTX):
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
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.blocks: list[str] = []
        self.counts: list[int] = []
        self.total_size = len(system_prompt)

    def add(self, content: str):
        assert CliLlmModel.MODEL is not None
        tokens = CliLlmModel.MODEL.tokenize(content.encode("utf-8"))
        tokens_count = len(tokens)

        while self.total_size > CliLlmModel.N_CTX and len(self.blocks) > 0:
            self.blocks.pop(0)
            self.total_size -= self.counts.pop(0)

        self.blocks.append(content)
        self.counts.append(tokens_count)
        self.total_size += tokens_count

    def get(self) -> str:
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


#! Command: /history ID
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


#! TODO: Command: /ask MESSAGE


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
        ]


# TODO: Module IO:
# This module allows the AI to write files, manage directories, move within, etc... We also make sure
# that theses commands are secure : The user will set a list of directories that are available for writing / reading.
# If any of theses commands are executed outside of the scope of the allowed directories, theses commands MUST fail and return
# an error message (NO EXCEPTION, just the error). By default the project directory is the current working directory

# TODO: IO commands:
# /open_file PATH
# /close_file (CURRENT)
# /move FROM TO
# /copy FROM TO
# /remove FROM TO
# /cat -> CAT of 256 tokens MAX of the current cursor position + display before + display after (AROUND)
# /write (CURRENT) (TEXT)
# /move_cursor FILE POS
# /cursor FILE
# /replace (BEGIN) (END) (TEXT)
# /replace_regex REGEX SUBSTITUTION
# /grep REGEX -> Lines containing matching regex

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
    GRAMMAR = None  # LlamaGrammar.from_string('root ::= "/"[a-z].* [END]\n')
    CONTEXT_HISTORY = None
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
            "--modules-json",
            type=str,
            help="Path to a JSON file containing a list of module names.",
        )
        parser.add_argument(
            "--initial-task",
            type=str,
            help="Specify the initial task to execute after kernel initialization.",
            required=True,
        )
        parsed_args = parser.parse_args(argv)

        # Load modules
        if parsed_args.modules_json:
            with open(parsed_args.modules_json, "r") as f:
                module_names = json.load(
                    f
                )  # TODO: Make a JSON file that globalize the parsed args (file config)
                if not isinstance(module_names, list) or not all(
                    isinstance(name, str) for name in module_names
                ):
                    raise ValueError("JSON file must contain a list of module names.")
        else:
            module_names = parsed_args.modules or []

        # Register modules
        for name in module_names:
            match name:
                case "core":
                    CliLlmKernel.MODULES.register(CliLlmModuleCore())

                case _:
                    raise ValueError(f"Module '{name}' is not recognized.")

        # Specify initial task
        CliLlmKernel.TASK = parsed_args.initial_task
        # TODO: Create TOT task instance with the initial task

    @staticmethod
    def get_wd() -> str:
        if CliLlmKernel.MODULES.is_registered("io"):
            return os.getcwd()
            # TODO: If os module is activated, then we return here the current directory
        else:
            return os.getcwd()

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
            "**Structured Reasoning Protocol**\n\n"
            "Follow this exact sequence for EVERY action:\n"
            "1. THINK: Analyze the current task and plan what you should do. It is useful to call these commands several times before going to 2. This allows you to test your thoughts, and check that the next action is actually what you want to do. Always check that your thinking is in line with the idea being developed.\n"
            "2. ACT: Choose ONE command from the available commands section below.\n"
            "3. REFLECT: Verify action results before continuing.\n"
            "4. PLAN: Set, plan, test and improve your thoughts. Are you getting the results you want? Is there room for improvement? Have you made mistakes?\n\n"
            "**Current Task Context**\n"
            f"Main Objective: {CliLlmKernel.TASK}\n"
            "**Execution Cycle Template**\n"
            "EXAMPLE 1 - Math Problem:\n"
            "/reason Analyzing equation: First I need to isolate x... [END]\n"
            "/create_task -c 12 -t Verify solution by substitution [END]\n"
            "/comment 13 Checking if x=5 satisfies original equation [END]\n\n"
            "EXAMPLE 2 - Programming Task:\n"
            "/reason The infinite loop likely comes from incorrect termination condition [END]\n"
            "/create_task -c 8 -t Implement new loop structure [END]\n"
            "/comment 9 Trying while loop with decrementing counter [END]\n"
            "/done 9 [END]\n\n"
            "**Strict Command Rules:**\n"
            "- ALWAYS use /reason BEFORE creating tasks\n"
            "- Create MAX 3 subtasks per action\n"
            "- Always terminate a command call with [END]! EXAMPLE: \\<command> <args or parameters...> [END]. One command call always finish with [END]!\n"
            "- Verify dependencies with /parents before /done\n"
            "- Use /comment after EVERY task creation\n\n"
            "**Available Commands:**\n"
            f"{CliLlmKernel.get_command_list_str()}\n"
            "</s> [/INST]"
        )

    @staticmethod
    def compute_prompt():
        assert CliLlmKernel.CONTEXT_HISTORY is not None
        return f"{CliLlmKernel.CONTEXT_HISTORY.get()}\n"

    @staticmethod
    def init_run():
        # System Prompt
        sys_prompt = CliLlmKernel.compute_sys_prompt()
        print(sys_prompt)
        print("")

        # Generate first answer
        full_response = CliLlmModel.send_prompt(
            sys_prompt,
            CliLlmKernel.GRAMMAR,
        )
        argv = full_response["choices"][0]["text"]
        argv = argv.strip()
        answer = f"\n{CliLlmKernel.get_wd()}$ /{argv} [END]\n"

        # Process first command
        argv_words = argv.split(" ")
        result = CliLlmKernel.COMMANDS.execute(argv_words[0], argv_words[1:])

        # Initialize History
        CliLlmKernel.CONTEXT_HISTORY = CliLlmMessageHistory(sys_prompt + answer)
        CliLlmKernel.CONTEXT_HISTORY.add("[INST]" + result + "[/INST]\n")

    @staticmethod
    def run():
        assert CliLlmKernel.CONTEXT_HISTORY is not None

        # Continuous Prompt
        prompt = CliLlmKernel.compute_prompt()

        # Appending working directory to generated prompt
        prompt += f"\n{CliLlmKernel.get_wd()}$ /"

        # Generate answer
        # TODO: Maybe do a while loop until we found the [END] token
        full_response = CliLlmModel.send_prompt(prompt, CliLlmKernel.GRAMMAR)
        argv = full_response["choices"][0]["text"]
        argv = argv.strip()
        answer = f"\n{CliLlmKernel.get_wd()}$ /{argv} [END]\n"
        print(answer)
        print("")

        # Process command
        argv_words = argv.split(" ")
        result = CliLlmKernel.COMMANDS.execute(argv_words[0], argv_words[1:])
        # print(CliLlmKernel.STATES.message_status())

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
