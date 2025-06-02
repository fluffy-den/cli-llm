from typing import Callable
from threading import Lock


class LazyAgentCallbackManager:
    def __init__(self) -> None:
        self._callbacks: list[Callable[..., None]] = []
        self._lock = Lock()

    def register(self, callback: Callable[..., None]) -> None:
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def unregister(self, callback: Callable[..., None]) -> None:
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def trigger(self, *args, **kwargs) -> None:
        with self._lock:
            callbacks_copy = list(self._callbacks)

        for callback in callbacks_copy:
            callback(*args, **kwargs)
