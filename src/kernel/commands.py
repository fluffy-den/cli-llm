from kernel.command import command
from kernel.states import states


class commands:
    def __init__(self):
        self.commands: dict[str, command] = {}

    def get_command_list(self) -> list[command]:
        return list(self.commands.values())

    def has_command(self, name: str) -> bool:
        return name in self.commands

    def get_command(self, name: str) -> command:
        if not self.has_command(name):
            raise ValueError(f"Command '{name}' not found")
        return self.commands[name]

    def register(self, commands: list[command]):
        for cmd in commands:
            self.commands[cmd.name] = cmd

    def execute(self, states: states, cmd: str, argv: list[str]) -> str:
        if cmd not in self.commands:
            return f"Command '{cmd}' not found. Please use `cmd.help -h help` to see available commands."
        return self.commands[cmd].action(states, argv)
