class LazyAgentModuleAbstract:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return f"LazyAgentModule(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()

    def run(self) -> None:
        raise NotImplementedError("The run method must be implemented by the subclass.")


class LazyAgentModuleManager:
    def __init__(self) -> None:
        self.modules = {}

    def _validate_module_name(self, name: str) -> None:
        if name not in self.modules:
            raise ValueError(f"Module '{name}' is not registered.")

    def register_module(self, module: LazyAgentModuleAbstract) -> None:
        if module.name in self.modules:
            raise ValueError(f"Module '{module.name}' is already registered.")
        self.modules[module.name] = module

        # TODO: Log the registration of the module

    def get_module(self, name: str) -> LazyAgentModuleAbstract:
        self._validate_module_name(name)
        return self.modules[name]

    def run_module(self, name: str) -> None:
        self._validate_module_name(name)
        self.modules[name].run()
