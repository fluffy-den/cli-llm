from typing import Any


###! States
class states:
    def __init__(self):
        self.states_data: dict[
            str, Any
        ] = {}  # Dictionary to store module-specific data

    def register(self, name: str, data: Any):
        if self.is_registered(name):
            raise ValueError("State already registered")
        self.states_data[name] = data

    def retrieve(self, name: str) -> Any:
        return self.states_data[name]

    def is_registered(self, name: str) -> bool:
        return name in self.states_data

    def message_status(self):
        return ""
