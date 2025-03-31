from kernel.states import states

from argparse import ArgumentParser


class command:
    def __init__(self, name: str):
        self.name = name

    def action(self, state: states, args: list[str]) -> str:
        raise NotImplementedError("This method must be implemented in a subclass")

    def short(self) -> str:
        raise NotImplementedError("This method must be implemented in a subclass")

    def long(self) -> str:
        raise NotImplementedError("This method must be implemented in a subclass")
