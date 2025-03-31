from llama_cpp import LlamaGrammar, Llama

from kernel.command_module import command_module
from kernel.states import states
from kernel.commands import commands
from kernel.config import config
from kernel.context_history import context_history

import os


class kernel:
    SHOULD_EXIT = False
    GRAMMAR = LlamaGrammar.from_string("root ::= [a-z].* [END]")
    CONTEXT_HISTORY = None
    COMMANDS = commands()
    STATES = states()
    MODEL = None
    VERSION = "v0.0.1"

    @staticmethod
    def get_current_wd() -> str:
        if kernel.STATES.is_registered("io"):
            return kernel.STATES.retrieve("io").working_directory
        else:
            return os.getcwd()

    @staticmethod
    def execute_command(cmd: str, argv: list[str]) -> str:
        return kernel.COMMANDS.execute(kernel.STATES, cmd, argv)

    @staticmethod
    def add_module(module: command_module):
        commands = module.retrieve_commands()
        state = module.retrieve_initial_state()

        kernel.COMMANDS.register(commands)
        kernel.STATES.register(module.name, state)

    @staticmethod
    def compute_num_threads():
        cpu_count = os.cpu_count()
        if cpu_count is None:
            cpu_count = 1
        else:
            cpu_count -= 2
        return cpu_count

    @staticmethod
    def init():
        # Model
        kernel.MODEL = Llama.from_pretrained(
            repo_id=config.MODEL,
            filename=config.MODEL_QUANT,
            n_ctx=config.N_CTX,
            n_temperature=config.N_TEMPERATURE,
            n_gpu_layers=config.N_GPU_LAYERS,
            n_batch=config.N_BATCH,
            n_threads=kernel.compute_num_threads(),
            offload_kqv=True,
            main_gpu=0,
            tensor_split=[0.9],
            use_mlock=True,
        )

        # Print Message
        print("")
        print("Pulsar Project - Auto LLM w/ Mistral")
        print(kernel.VERSION)
        print("")

    @staticmethod
    def compute_sys_prompt():
        return (
            "[INST] <s>\n"
            "**COMMAND EXECUTION PROTOCOL**\n"
            "**DIRECTIVE**: You are an autonomous command-line interface agent that exclusively communicates through\n"
            "system commands. You will logically subdivide complex tasks into subtasks, think about them, test your\n"
            "ideas then try to implement them. You will be able to execute commands inside the project directory,\n"
            "and some outside too. If something is wrong or incorrect, try to fix it by thinking differently, by\n"
            "seeking for another path of thoughts.\n"
            "\n"
            "**CORE MECHANISM**\n"
            "   - ALL outputs have the form: `$ COMMAND PARAM_1, PARAM_2, ..., PARAM_N [END]`\n"
            "     Where `PARAM_i` is the i-th parameter of the command. Some commands doesn't need any parameter.\n"
            "     Write your commands accordingly.\n"
            "   - ALL outputs must finish with: `[END]` You are allowed to write your command / parameters as long as\n"
            "     needed, but the `[END]` is mandatory.\n"
            "\n"
            "**HELP COMMAND**\n"
            "   - The help command is the most important command. It allows you to understand which commands you can\n"
            "     run to perform tasks.\n"
            "   - Write `$ cmd.help [END]` to print the help message. It will help you understand what you can do and\n"
            "     have to do.\n"
            "\n"
            "**EXECUTION FLOW**\n"
            "To solve your corresponding task, you have to split theses tasks into sub-tasks. Here is the chain of\n"
            "thoughts to solve the tasks.\n"
            "   1. Write down your ideas, reason and hypothesis about the current task. Generate sub-tasks everytime.\n"
            "   2. Test logically your ideas. If something seems impossible, reject the idea.\n"
            "   3. If your idea seems logical, try to implement it. If not, go back to 1.\n"
            "   4. With the implementation, test the implementation logically and Normaly. If their is a logical error\n"
            "      Reimplement it by going back to 3.\n"
            "   5. At this point, the implementation must be done. Write unit tests about this task.\n"
            "   6. Run the unit tests. If they fail, go back to 3. If they pass, it's correct.\n"
            "   7. Review your implementation. Why does it implement the given task? What can be improved?\n"
            "   8. At this point, the task is done. You can now move to the next task. If their is no more tasks,\n"
            "      generate new ones. Go back to 1 when it's done.\n"
            "\n"
            "**INITIAL TASK**\n"
            f"{config.INITIAL_TASK}\n"
            "\n"
            "**RULES:**"
            "   1. You NEVER describe what a command does. The system will do it for you.\n"
            "   2. You NEVER simulate command output.\n"
            "   3. You ONLY generate one command at a time.\n"
            "   4. You NEVER explain your thought process. Just generate the command. If you want to write your\n"
            "      thougths, it exists some commands that take in parameter your thoughts, but ONLY use them.\n"
            "      write `cmd.help -h help [END]` to see them.\n"
            "\n"
            "**EXEMPLES OUTPUTS**\n"
            "$ cmd.help [END]\n"
            "$ cmd.help -h help [END]\n"
            "\n"
            "</s> [/INST]"
        )

    @staticmethod
    def compute_prompt():
        assert kernel.CONTEXT_HISTORY is not None
        return f"{kernel.CONTEXT_HISTORY.get()}\n{kernel.STATES.message_status()}"

    @staticmethod
    def compute_answer(prompt: str):
        assert kernel.MODEL is not None
        return kernel.MODEL.create_completion(
            prompt=prompt,
            max_tokens=config.N_CTX - len(prompt) - 32,
            temperature=config.N_TEMPERATURE,
            top_p=config.N_TOP_P,
            top_k=config.N_TOP_K,
            min_p=config.N_MIN_P,
            stop=["[END]"],
            grammar=kernel.GRAMMAR,
        )

    @staticmethod
    def init_run():
        # System Prompt
        sys_prompt = kernel.compute_sys_prompt()
        print(sys_prompt)
        print("")

        # Appending working directory to generated prompt
        sys_prompt += f"\n{kernel.get_current_wd()}$ cmd."

        # Generate first answer
        full_response = kernel.compute_answer(sys_prompt)
        argv = full_response["choices"][0]["text"]
        answer = f"\n{kernel.get_current_wd()}$ cmd.{argv}"
        print(answer)
        print("")

        # Process first command
        argv_words = argv.split(" ")
        result = kernel.COMMANDS.execute(kernel.STATES, argv_words[0], argv_words[1:])
        print(kernel.STATES.message_status())

        # Initialize History
        kernel.CONTEXT_HISTORY = context_history(sys_prompt + argv)
        kernel.CONTEXT_HISTORY.add("[INST]" + result + "[/INST]\n")

    @staticmethod
    def run():
        assert kernel.CONTEXT_HISTORY is not None

        # Continuous Prompt
        prompt = kernel.compute_prompt()
        print(prompt)
        print("")

        # Appending working directory to generated prompt
        prompt += f"\n{kernel.get_current_wd()}$ cmd."

        # Generate answer
        full_response = kernel.compute_answer(prompt)
        argv = full_response["choices"][0]["text"]
        answer = f"\n{kernel.get_current_wd()}$ cmd.{argv}"
        print(answer)
        print("")

        # Process command
        argv_words = argv.split(" ")
        result = kernel.COMMANDS.execute(kernel.STATES, argv_words[0], argv_words[1:])
        print(kernel.STATES.message_status())

        # Update history
        kernel.CONTEXT_HISTORY.add(prompt + argv + "[INST]" + result + "[/INST]\n")

    @staticmethod
    def thinking_loop():
        kernel.init()
        kernel.init_run()
        while not kernel.SHOULD_EXIT:
            kernel.run()
