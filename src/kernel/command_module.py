from kernel.command import command

from typing import Any


class command_module:
    def __init__(self, name: str):
        self.name = name

    def retrieve_commands(self) -> list[command]:
        raise NotImplementedError("register_commands() not implemented")

    def retrieve_initial_state(self) -> Any:
        raise NotImplementedError("register_state() not implemented")
