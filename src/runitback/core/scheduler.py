"""Model-access scheduling shared by every sport's server (one CPU, many requests)."""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator


class ModelScheduler:
    """Serialize model access while giving interactive requests priority."""

    def __init__(self):
        self._condition = threading.Condition()
        self._active = False
        self._interactive_waiters = 0

    @contextmanager
    def interactive(self) -> Iterator[None]:
        with self._condition:
            self._interactive_waiters += 1
            try:
                while self._active:
                    self._condition.wait()
                self._active = True
            finally:
                self._interactive_waiters -= 1
        try:
            yield
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()

    @contextmanager
    def background(self) -> Iterator[None]:
        with self._condition:
            while self._active or self._interactive_waiters:
                self._condition.wait()
            self._active = True
        try:
            yield
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()


# One scheduler for the whole process: every sport's models share the same CPU.
MODEL_SCHEDULER = ModelScheduler()
