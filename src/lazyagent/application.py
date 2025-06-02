from enum import Enum
from threading import Lock
from datetime import datetime
from typing import Callable
from utility import LazyAgentCallbackManager


class LazyAgentApplicationStates(Enum):
    CORE_RUNNING = "core_running"
    CORE_STOPPED = "core_stopped"


class LazyAgentApplicationState:
    STATE = LazyAgentApplicationStates.CORE_RUNNING

    _LOCK = Lock()

    @staticmethod
    def set_state(state: LazyAgentApplicationStates) -> None:
        with LazyAgentApplicationState._LOCK:
            LazyAgentApplicationState.STATE = state

    @staticmethod
    def get_state() -> LazyAgentApplicationStates:
        with LazyAgentApplicationState._LOCK:
            return LazyAgentApplicationState.STATE

    @staticmethod
    def is_running() -> bool:
        with LazyAgentApplicationState._LOCK:
            return (
                LazyAgentApplicationState.STATE
                == LazyAgentApplicationStates.CORE_RUNNING
            )


class LazyAgentLogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LazyAgentLogger:
    def _get_timestamp_str(self) -> str:
        now = datetime.now()
        return now.strftime("%Y-%m-%dT%H:%S.%fZ")

    def _get_formatted_str(self, level: LazyAgentLogLevel, message: str) -> str:
        ts = self._get_timestamp_str()
        return f"[{ts}] [{level.value}] {message}"

    def __init__(self, logging_path: str):
        self.logging_path = logging_path
        self.logging_file = open(logging_path, "a", encoding="utf-8")
        if not self.logging_file:
            raise ValueError(f"Failed to open log file at {logging_path}")

        self.logging_callbacks = LazyAgentCallbackManager()

        self._lock = Lock()

    def add_callback(self, callbacks: Callable) -> None:
        self.logging_callbacks.append(callbacks)

    def log(self, level: LazyAgentLogLevel, message: str) -> None:
        entry = self._get_formatted_str(level, message)

        with self._lock:
            try:
                self.logging_file.write(entry + "\n")
                self.logging_file.flush()
            except Exception as e:
                print(f"Failed to write log entry: {e}")


class LazyAgentConfig:
    pass
