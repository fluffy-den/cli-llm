from argparse import ArgumentError, ArgumentParser
from typing import Any

from numpy import add

from kernel.command import command
from kernel.command_module import command_module
from kernel.kernel import kernel
from kernel.states import states


###! Help
class module_core_cmd_help(command):
    def __init__(self):
        self.arg_parser = ArgumentParser(add_help=False)
        self.arg_parser.add_argument("-h", type=str, required=False, nargs="?")

        super().__init__("help")

    def action(self, state: Any, args: list[str]) -> str:
        if len(args) == 0 or args[0] == "":
            return self.short()

        try:
            self.arg_parser.parse_args(args)
        except ArgumentError as e:
            return e.message

        if self.arg_parser.help:
            subcommand = self.arg_parser.help
            if kernel.COMMANDS.has_command(subcommand):
                return kernel.COMMANDS.get_command(subcommand).long()

            return "Error: Command not found for ${subcommand}"

        else:
            return self.short()

    def short(self) -> str:
        return (
            "\ncmd.help:\n"
            "   A call to this command will display this help message. Please write\n"
            "   - `cmd.help -h [COMMAND] [END]` to get more information about a specific command.\n"
            "   - `cmd.help -h help [END]` to get a list of available commands.\n"
            "   NOTE: When specifying a command, [COMMAND] is the name of the command in\n"
            "   $cmd.[COMMAND].\n"
        )

    def long(self) -> str:
        commands = kernel.COMMANDS.get_command_list()

        return f"*** AVAILABLE COMMANDS: ***\n\t{' '.join(rf'$cmd.{cmd}' for cmd in commands)}"


###! Exit
class module_core_cmd_exit(command):
    def __init__(self):
        args = ArgumentParser()
        super().__init__("exit")


class module_core(command_module):
    def __init__(self):
        super().__init__("core")

    def retrieve_commands(self) -> list[command]:
        return [module_core_cmd_help(), module_core_cmd_exit()]

    def retrieve_initial_state(self) -> Any:
        return None
