from collections import deque
from typing import Any, Deque, Dict, Iterable, Optional

from .io_port import IOPort


class InMemoryIOPort(IOPort):
    """A minimal in-memory port for tests and wiring examples.

    - Maintains a FIFO queue of (message, meta) pairs.
    - ack/nack update bookkeeping only; messages stay delivered to caller.
    - Not thread-safe; intended for single-threaded tests.
    """

    def __init__(self, maxlen: Optional[int] = None) -> None:
        self._queue: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self._acked: set[str] = set()
        self._nacked: set[str] = set()

    def send(self, message: Any, meta: Dict[str, Any]) -> None:
        payload = {"message": message, "meta": meta}
        self._queue.append(payload)

    def receive(self, meta: Dict[str, Any]) -> Iterable[Any]:
        while self._queue:
            yield self._queue.popleft()

    def ack(self, message_id: str) -> None:
        self._acked.add(message_id)
        self._nacked.discard(message_id)

    def nack(self, message_id: str, reason: str) -> None:
        self._nacked.add(message_id)
        self._acked.discard(message_id)

    def flush(self) -> None:
        self._queue.clear()

    def attach(self, adapter: Any) -> None:
        # No-op for in-memory; adapters would wrap external transports.
        return None

    @property
    def acked(self) -> set[str]:
        return self._acked

    @property
    def nacked(self) -> set[str]:
        return self._nacked
