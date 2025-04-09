from llama_cpp import LlamaGrammar, Llama

import datetime
import os
import sys
import argparse
import json
import textwrap


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
    N_TEMPERATURE = 0.65
    ## Top K
    N_TOP_K = 40
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
class CliLlmCommandExit(CliLlmCommandBase):
    def __init__(self):
        super().__init__("exit")

    def action(self, args: list[str]) -> str:
        CliLlmKernel.SHOULD_EXIT = True
        return "Exiting request has been asked...\n"

    def desc_short(self) -> str:
        return (
            "Exit the program. IMPORTANT: Don't use it unless you're sure that your tasks are finished, and"
            " that you can't continue further!\n"
        )

    def desc(self) -> str:
        return self.desc_short()


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
    def __init__(self, task_id: int, title: str, dependencies: list[int] | None = None):
        self.id = task_id
        self.title = title
        self.dependencies = dependencies if dependencies is not None else []
        self.thoughts: list[str] = []
        self.status = CliLlmTOTTaskStatus.PENDING
        self.created_at = datetime.datetime.now()
        self.updated_at = datetime.datetime.now()


class CliLlmTOTTaskManager:
    TASKS: dict[int, CliLlmTOTTask] = {}
    CURRENT_TASK_ID: int | None = None  # Track active task

    @classmethod
    def generate_task_id(cls) -> int:
        return max(cls.TASKS.keys(), default=0) + 1

    @classmethod
    def add_task(cls, task: CliLlmTOTTask):
        if task.id in cls.TASKS:
            raise ValueError(f"Task ID {task.id} already exists")
        cls.TASKS[task.id] = task
        cls.CURRENT_TASK_ID = task.id  # Set as current task

    @classmethod
    def update_task_status(cls, task_id: int, status: int):
        task = cls.TASKS.get(task_id)
        if task:
            task.status = status
            task.updated_at = datetime.datetime.now()
        else:
            raise ValueError(f"Task {task_id} not found")

    @classmethod
    def get_task(cls, task_id: int) -> CliLlmTOTTask | None:
        return cls.TASKS.get(task_id)

    @classmethod
    def get_child_tasks(cls, task_id: int) -> list[CliLlmTOTTask]:
        return [t for t in cls.TASKS.values() if task_id in t.dependencies]

    @classmethod
    def validate_dependencies(cls, dependencies: list[int]) -> bool:
        return all(dep in cls.TASKS for dep in dependencies)


class CliLlmCommandreason(CliLlmCommandBase):
    def __init__(self):
        super().__init__("reason")

    def action(self, args: list[str]) -> str:
        try:
            task_id = None
            message = []
            i = 0

            while i < len(args):
                if args[i] == "-d" and i + 1 < len(args):
                    if args[i + 1].isdigit():
                        task_id = int(args[i + 1])
                        i += 2
                    else:
                        return "Invalid task ID after -d"
                else:
                    message.append(args[i])
                    i += 1

            if not message:
                return "Error: Empty thought"

            thought = " ".join(message)

            if task_id is not None:
                task = CliLlmTOTTaskManager.TASKS.get(task_id)
                if not task:
                    return f"Task {task_id} not found"
                task.thoughts.append(thought)
                return f"Added thought to task {task_id}"
            else:
                # Add to global history
                return "Added global thought"

        except Exception as e:
            return f"Error recording thought: {str(e)}"

    def desc(self) -> str:
        return (
            "Usage: /reason {-d (TASK_ID)} (MESSAGE) [END]\n"
            "Let you reason, validate, explore about a task (TASK_ID) or the last task being added. (MESSAGE) can be a very long string message. Use this command to develop the reason of a task! You can use maths, code of any language, formulas...\n"
            "   (optional) -d (TASK_ID): This optional argument allows you to specify on which (TASK_ID) you're reason on.\n"
            "Examples:\n"
            "/reason I think that I should begin with X to do Y, but I must also consider the effects of Z on X... Maybe Y implies X... Etc... [END]\n"
            "/reason -d 5 The task '5' might involve ... Let's try to write the formula... Is x + y = z... [END]\n"
            "/reason Wait 2 + 3 isn't equal to 6! [END]\n"
            "/reason Is it right that X ?... [END]\n"
        )


class CliLlmCommandreasonNew(CliLlmCommandBase):
    def __init__(self):
        super().__init__("task_new")

    def action(self, args: list[str]) -> str:
        try:
            # Parse arguments
            task_name_parts = []
            dependencies = []
            parse_deps = False

            for arg in args:
                if arg == "-d":
                    parse_deps = True
                elif parse_deps:
                    if arg.isdigit():
                        dependencies.append(int(arg))
                    else:
                        return f"Invalid dependency: {arg} must be integer"
                else:
                    task_name_parts.append(arg)

            if not task_name_parts:
                return "Error: Task name required"

            # Validate dependencies
            if not CliLlmTOTTaskManager.validate_dependencies(dependencies):
                return "Error: One or more dependencies don't exist"

            # Create task
            new_id = CliLlmTOTTaskManager.generate_task_id()
            new_task = CliLlmTOTTask(
                task_id=new_id,
                title=" ".join(task_name_parts),
                dependencies=dependencies,
            )
            CliLlmTOTTaskManager.add_task(new_task)
            return f"Created task with ID: {new_id}"

        except Exception as e:
            return f"Error creating task: {str(e)}"

    def desc(self) -> str:
        return (
            "Usage: /task_new (TASK_NAME) {-d (DEPENDENCY_ID_1) (DEPENDENCY_ID_2) ...} [END]\n"
            "Creates a new task with the specified name and dependencies.\n"
            "   (optional) -d (DEPENDENCY_1) (DEPENDENCY_2): This optional argument allows you to specify that this new task is a subtask of every"
            " dependencies specified by '-d'. You can specify as many dependencies as you want.\n"
            "Note: A task must not be done or failed when reason about it!\n"
            "Examples:\n"
            "/task_new Discovering project files... [END]\n"
            "The previous command will return a TASK_ID, which you can use to refer to this task.\n"
            "/task_new -d 5 Discovering files of folder X...[END]\n"
        )


class CliLlmCommandreasonTask(CliLlmCommandBase):
    def __init__(self):
        super().__init__("task")

    def action(self, args: list[str]) -> str:
        try:
            task_id = None
            task = None
            flags = set()
            i = 0

            # Parse arguments
            while i < len(args):
                if args[i] == "-t" and i + 1 < len(args):
                    task_id = int(args[i + 1])
                    i += 2
                elif args[i].startswith("-"):
                    flags.add(args[i][1:])
                    i += 1
                else:
                    i += 1

            # Resolve task ID
            if not task_id and "i" in flags:
                task_id = CliLlmTOTTaskManager.CURRENT_TASK_ID

            if task_id is not None:
                task = CliLlmTOTTaskManager.TASKS.get(task_id)
                if not task:
                    return f"Task {task_id} not found"

            # Build response
            response = []
            if "p" in flags:  # Parents
                if not task:
                    return "Error: Task required to get parent tasks"

                parents = [
                    f"{dep}: {CliLlmTOTTaskManager.TASKS[dep].title}"
                    for dep in task.dependencies
                    if dep in CliLlmTOTTaskManager.TASKS
                ]
                response.append(f"Parents: {', '.join(parents) or 'None'}")

            if "c" in flags:  # Children
                if not task_id:
                    return "Error: Task required to get child tasks"
                children = [
                    f"{t.id}: {t.title}"
                    for t in CliLlmTOTTaskManager.get_child_tasks(task_id)
                ]
                response.append(f"Children: {', '.join(children) or 'None'}")

            if "s" in flags:  # Status
                if not task:
                    return "Error: Task required to get status"
                response.append(f"Status: {CliLlmTOTTaskStatus.to_string(task.status)}")

            if "h" in flags:  # Thoughts
                if not task:
                    return "Error: Task required to get thoughts"
                thoughts = "\n".join(
                    [f"{i + 1}. {t}" for i, t in enumerate(task.thoughts)]
                )
                response.append(f"Thoughts:\n{thoughts or 'None'}")

            return "\n".join(response) or "No information requested"

        except Exception as e:
            return f"Error retrieving task info: {str(e)}"

    def desc(self) -> str:
        return (
            "Usage: /task {-t (TASK_ID)} {-i} {-p} {-c} [END]\n"
            "This command allows you to get information about a specific task.\n"
            "   (optional) -t (TASK_ID): This optional argument allows you to execute the others arguments specific to this (TASK_ID).\n"
            "   (optional) -i: Get the (TASK_ID) of the current running task. Irrevelant when using with '-t'\n"
            "   (optional) -p: Get a list of the parent tasks of this task or the specified task with '-t'\n"
            "   (optional) -c: Get a list of the child tasks of this task or the specified task with '-t'\n"
            "   (optional) -d: Get the creation time of the specified task.\n"
            "   (optional) -u: Get the update time of the specified task.\n"
            "   (optional) -s: Get the status of the specified task.\n"
            "   (optional) -h: Get the thoughts history about the specified task.\n"
            "Note: You can combine arguments together.\n"
            "Examples:"
            "/task -i [END]\n"
            "/task -t 5 -p [END]\n"
            "/task -t 7 -c [END]\n"
            "/task -t 9 -p -c [END]\n"
            "/task -t 10 -d -u -s -h [END]\n"
        )


class CliLlmCommandreasonDone(CliLlmCommandBase):
    def __init__(self):
        super().__init__("task_done")

    def action(self, args: list[str]) -> str:
        if len(args) != 1 or not args[0].isdigit():
            return "Usage: /task_done (TASK_ID) [END]"

        task_id = int(args[0])
        try:
            CliLlmTOTTaskManager.update_task_status(
                task_id, CliLlmTOTTaskStatus.SUCCESS
            )
            return f"Task {task_id} marked successful"
        except Exception as e:
            return f"Error updating task: {str(e)}"

    def desc(self) -> str:
        return (
            "Usage: /task_done (TASK_ID) [END]\n"
            "Marks the specified task as done.\n"
            "Example:\n"
            "/task_done 24 [END]\n"
        )


class CliLlmCommandreasonFail(CliLlmCommandBase):
    def __init__(self):
        super().__init__("task_fail")

    def action(self, args: list[str]) -> str:
        if len(args) != 1 or not args[0].isdigit():
            return "Usage: /task_fail TASK_ID [END]"

        task_id = int(args[0])
        try:
            CliLlmTOTTaskManager.update_task_status(task_id, CliLlmTOTTaskStatus.FAILED)
            return f"Task {task_id} marked failed"
        except Exception as e:
            return f"Error updating task: {str(e)}"

    def desc(self) -> str:
        return (
            "Usage: /task_fail (TASK_ID) [END]\n"
            "Marks the specified task as failed.\n"
            "Example:\n"
            "/task_fail 3 [END]\n"
        )


# TODO: Move to new module 'interactive' this command
class CliLlmCommandreasonAsk(CliLlmCommandBase):
    def __init__(self):
        super().__init__("ask")

    def action(self, args: list[str]) -> str:
        # TODO: Allow AI to ask a a question to the user
        return super().action(args)

    def desc(self) -> str:
        return (
            "Usage: /ask (QUESTION) [END]\n"
            "Allows the AI to ask a question to the user. This will create a special task called\n"
        )


class CliLlmModuleCore(CliLlmModuleBase):
    def __init__(self):
        super().__init__("core")

    def retrieve_commands(self) -> list[CliLlmCommandBase]:
        return [
            # TODO: Rework TOT: Less, too difficult for AI to understand. Better :
            # /exit
            # /reason ABOUT SOMETHING...
            # /task_create TITLE -d 0,1,2
            # /task_comment ID MESSAGE
            # /task_done ID
            # /task_fail ID
            # /task_pending ID
            # /task_current ID
            # /task_childrens ID
            # /task_parents ID
            # /task_status  ID
            # /task_history ID
            # /ask MESSAGE
            #
            CliLlmCommandExit(),
            CliLlmCommandreason(),
            CliLlmCommandreasonTask(),
            CliLlmCommandreasonNew(),
            CliLlmCommandreasonDone(),
            CliLlmCommandreasonFail(),
        ]


# TODO: Module IO:
# This module allows the AI to write files, manage directories, move within, etc... We also make sure
# that theses commands are secure : The user will set a list of directories that are available for writing / reading.
# If any of theses commands are executed outside of the scope of the allowed directories, theses commands MUST fail and return
# an error message (NO EXCEPTION, just the error). By default the project directory is the current working directory

# TODO: IO commands:
# /file_open PATH
# /file_close (CURRENT)
# /file_info -> Cursor position, file size, CAT of 256 tokens of the current cursor position (AROUND)
# /file_write (CURRENT) (TEXT)
# /file_move_cursor FILE POS
# /file_cursor FILE
# /file_replace (BEGIN) (END) (TEXT)
# /file_replace_regex REGEX SUBSTITUTION
# /file_grep REGEX -> Lines containing matching regex

# TODO: Module Coding Tools:
# This module allows the AI to use LSP tools such as treesitter, but also grep

# TODO: Coding Tools
# /code_symbols FILE
# /workspace_symbols FILE
# /code_grep_symbols FILE REGEX
# /workspace_grep_symbols FILE REGEX
# /replace_symbol SYMBOL NEW_SYMBOL

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
            "[INST] <s> "
            "**Advanced Reasoning Framework**  \n"
            '"You are an advanced AI trained to solve tasks methodically. Follow this structured framework to ensure rigorous, accurate, and human-like reasoning. Never skip steps unless explicitly instructed."  \n\n'
            "**Initial Task**:" + CliLlmKernel.TASK + "\n\n"
            "**Core Protocol**:  \n"
            "1. **Problem Analysis**:  \n"
            "   - **Clarify the Task**:  \n"
            "     - For each thoughts, use the command `\\reason MESSAGE [END]` to write down your thinkings. \n"
            "     - Paraphrase the problem in your own words.  \n"
            "     - Identify key objectives, constraints, and implicit requirements.  \n"
            "     - Example:  \n"
            '       Task: "Calculate the total cost of 5 apples priced at $2 each with a 10% discount."  \n'
            '       Clarification: "Objective: Compute final price after discount. Constraints: 5 apples, $2 per apple, 10% off total. Implicit: No taxes or fees."  \n\n'
            "   - **Domain Identification**:  \n"
            "     - Categorize the task (e.g., math, coding, creative writing, logic puzzle). And give a title of each task using `\\task_new TITLE [END]`\n"
            "     - Activate domain-specific reasoning patterns (e.g., PEMDAS for math, pseudocode for programming).  \n\n"
            "2. **Decomposition**:  \n"
            "   - **Break into Subproblems**:  \n"
            "     - Split the task into sequential, manageable steps.  \n"
            "     - Example for Coding:  \n"
            "       - Subproblem 1: Validate user input.  \n"
            "       - Subproblem 2: Implement algorithm X.  \n"
            "       - Subproblem 3: Handle edge cases (e.g., empty inputs).  \n\n"
            "   - **Prioritize Dependencies**:  \n"
            "     - Order subproblems logically (e.g., solve equation A before plotting graph B).  \n\n"
            "3. **Hypothesis Generation**:  \n"
            "   - **Brainstorm Approaches**:  \n"
            "     - Propose 2-3 potential methods to solve each subproblem.  \n"
            "     - Example for Math:  \n"
            "       - Method 1: Apply linear algebra.  \n"
            "       - Method 2: Use trial-and-error with constraints.  \n\n"
            "   - **Select Optimal Strategy**:  \n"
            "     - Choose the method that balances efficiency, accuracy, and resource limits.  \n"
            '     - Rationale: "Linear algebra scales better than trial-and-error for large datasets."  \n\n'
            "4. **Step-by-Step Execution**:  \n"
            "   - **Detailed Walkthrough**:  \n"
            "     - Explain every operation, formula, or logical deduction.  \n"
            "     - For Creative Tasks:  \n"
            '       - "Introduce a protagonist with trait X to create conflict Y, aligning with theme Z."  \n\n'
            "   - **Inline Verification**:  \n"
            "     - After each step, validate outputs before proceeding:  \n"
            '       - Arithmetic: "5 apples × 2 = 10 → 10 × 0.9 = 9 ✔️"  \n'
            '       - Logic: "If X is true, then Y must follow because..."  \n\n'
            "5. **Error Checking & Recovery**:  \n"
            "   - **Self-Verification Checklist**:  \n"
            "     - Cross-validate results using:  \n"
            "       - Alternative Methods: Solve the same problem a different way.  \n"
            "       - Unit Tests: For code, test edge cases (e.g., zero, null, extremes).  \n"
            '       - External Knowledge: "PrimeGrid’s database confirms 1,000,000th prime is 15,485,863."  \n\n'
            "   - **Iterative Refinement**:  \n"
            "     - If errors are found:  \n"
            '       - Diagnose root cause (e.g., "Miscount in loop iteration 3").  \n'
            "       - Correct and re-verify the entire chain.  \n\n"
            "6. **Ambiguity Handling**:  \n"
            "   - **Explicit Assumptions**:  \n"
            "     - Document any assumptions made due to incomplete data.  \n"
            '     - Example: "Assuming ‘price’ refers to USD and discounts apply post-tax."  \n\n'
            "   - **Explore Scenarios**:  \n"
            "     - Provide answers for multiple interpretations if ambiguity persists:  \n"
            '       - Scenario 1: "If X is true, then Answer = A."  \n'
            '       - Scenario 2: "If Y is true, then Answer = B."  \n\n'
            "7. **Final Answer Synthesis**:  \n"
            "   - **Unified Response**:  \n"
            "     - Consolidate results into a clear, concise answer.  \n"
            "     - Format based on task type:  \n"
            "       - Math: Boxed answer \\boxed{9}.  \n"
            "       - Code: Full script with comments.  \n"
            "       - Creative: Story with marked climax/resolution.  \n\n"
            "   - **Lessons Learned**:  \n"
            '     - Summarize key insights (e.g., "Trig substitution simplified integration").  \n\n'
            "8. **Post-Solution Reflection**:  \n"
            "   - **Update tasks status with the commands `\\task_done`, `\\task_fail` and check other tasks status with `\\task`. \n"
            "   - **Critical Analysis**:  \n"
            "     - Evaluate your own performance:  \n"
            '       - "Strengths: Thorough decomposition. Weaknesses: Overlooked edge case Z."  \n\n'
            "   - **Improvement Plan**:  \n"
            '     - Suggest how to enhance future reasoning (e.g., "Study modular arithmetic for similar problems").  \n\n'
            "9. **Ending the Task**:  \n"
            "   - **Check if all tasks are done using the command `\\task_pending`. If no more task is pending, then you can exit the program using the command `\\exit`**  \n"
            "**Commands**:" + CliLlmKernel.get_command_list_str() + "\n" + "\n"
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
            sys_prompt + f"\n{CliLlmKernel.get_wd()}$ /", CliLlmKernel.GRAMMAR
        )
        argv = full_response["choices"][0]["text"]
        argv = argv.strip()
        answer = f"\n{CliLlmKernel.get_wd()}$ /{argv} [END]\n"
        print(answer)
        print("")

        # Process first command
        argv_words = argv.split(" ")
        result = CliLlmKernel.COMMANDS.execute(argv_words[0], argv_words[1:])
        # print(CliLlmKernel.STATES.message_status())

        # Initialize History
        CliLlmKernel.CONTEXT_HISTORY = CliLlmMessageHistory(sys_prompt + answer)
        CliLlmKernel.CONTEXT_HISTORY.add("[INST]" + result + "[/INST]\n")

    @staticmethod
    def run():
        assert CliLlmKernel.CONTEXT_HISTORY is not None

        # Continuous Prompt
        prompt = CliLlmKernel.compute_prompt()
        print(prompt)
        print("")

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
