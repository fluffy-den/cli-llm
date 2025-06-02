from modules import LazyAgentModuleAbstract

from textual.app import App, ComposeResult


class LazyAgentTUIApp(App):
    pass


class LazyAgentModuleTUI(LazyAgentModuleAbstract):
    def __init__(self) -> None:
        super().__init__("TUI")

        self.app = LazyAgentTUIApp()

    def run(self) -> None:
        self.app.run()
