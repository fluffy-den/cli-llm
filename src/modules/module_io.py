from kernel.command_module import command_module
from kernel.command import command
from kernel.kernel import kernel

from typing import Any


class state_io:
    working_directory = kernel.get_current_wd()


class module_io(command_module):
    def __init__(self):
        super().__init__("io")

    def retrieve_commands(self) -> list[command]:
        return []

    def retrieve_initial_state(self) -> Any:
        return state_io
