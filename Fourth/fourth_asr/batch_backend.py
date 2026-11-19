"""Synchronous Fourth session backend shared by concurrent HTTP request threads."""
from collections import deque
from concurrent.futures import Future
import math
import threading
import time


class BatchBackend:
    """Load and use one engine on one owner thread; callers wait independently.

    The HTTP service limits admitted calls and serializes each meeting. Distinct
    meetings may enqueue at the same time, including optional recovery decodes.
    """
    def __init__(self, backend_factory, *, max_batch_size=4, batch_wait_ms=2.0):
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if not math.isfinite(batch_wait_ms) or batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be nonnegative and finite")
        self.max_batch_size = max_batch_size
        self.batch_wait_ms = batch_wait_ms
        self._condition = threading.Condition()
        self._queue = deque()
        self._closing = False
        self._batch_count = 0
        self._largest_batch = 0
        ready = Future()
        self._thread = threading.Thread(target=self._run, args=(backend_factory, ready),
                                        name="fourth-http-model", daemon=True)
        self._thread.start()
        ready.result()

    def _submit(self, messages, audios, prefix):
        future = Future()
        with self._condition:
            if self._closing:
                raise RuntimeError("Model worker is closed")
            self._queue.append((time.monotonic(), (messages, audios, prefix), future))
            self._condition.notify()
        return future.result()

    def generate(self, messages, audios):
        return self._submit(messages, audios, "")

    def generate_with_prefix(self, messages, audios, prefix):
        if not prefix:
            raise ValueError("An assistant prefix is required")
        return self._submit(messages, audios, prefix)

    def _next_batch(self):
        with self._condition:
            self._condition.wait_for(lambda: self._queue or self._closing)
            if not self._queue:
                return []
            deadline = self._queue[0][0] + self.batch_wait_ms / 1000
            while not self._closing and len(self._queue) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return [self._queue.popleft() for _ in range(min(self.max_batch_size, len(self._queue)))]

    def _run(self, factory, ready):
        try:
            backend = factory()
            self.backend_name = type(backend).__name__
            self.max_new_tokens = getattr(backend, "max_new_tokens", None)
            self.max_context_tokens = getattr(backend, "max_context_tokens", None)
            self.max_audio_turns = getattr(backend, "max_audio_turns", None)
            self.allowed_tokens = getattr(backend, "allowed_tokens", None)
            self.enable_prefix_caching = getattr(backend, "enable_prefix_caching", False)
            batch_fn = getattr(backend, "generate_batch", None)
            if batch_fn is None:
                self.max_batch_size = 1
            else:
                self.max_batch_size = min(self.max_batch_size, getattr(backend, "max_batch_size", self.max_batch_size))
            if self.max_batch_size == 1:
                self.batch_wait_ms = 0.0
        except BaseException as exc:
            ready.set_exception(exc)
            return
        ready.set_result(None)
        try:
            while batch := self._next_batch():
                try:
                    if batch_fn is not None:
                        replies = batch_fn([call for _, call, _ in batch])
                        executed = getattr(backend, "last_batch_size", len(batch))
                    else:
                        _, (messages, audios, prefix), _ = batch[0]
                        reply = (backend.generate_with_prefix(messages, audios, prefix) if prefix
                                 else backend.generate(messages, audios))
                        replies, executed = [reply], 1
                    if len(replies) != len(batch):
                        raise RuntimeError("Backend returned the wrong number of meeting replies")
                    if executed:
                        with self._condition:
                            self._batch_count += 1
                            self._largest_batch = max(self._largest_batch, executed)
                except BaseException as exc:
                    for _, _, future in batch:
                        future.set_exception(exc)
                    if not isinstance(exc, Exception):
                        raise
                    continue
                for reply, (_, _, future) in zip(replies, batch):
                    if isinstance(reply, Exception):
                        future.set_exception(reply)
                    else:
                        future.set_result(reply)
        finally:
            with self._condition:
                self._closing = True
                while self._queue:
                    _, _, future = self._queue.popleft()
                    future.set_exception(RuntimeError("Model worker stopped"))
            if hasattr(backend, "close"):
                backend.close()

    def stats(self):
        """Host-side generate calls; vLLM schedules its GPU sequences internally."""
        with self._condition:
            return {"model_batches": self._batch_count, "largest_batch": self._largest_batch,
                    "queued_decodes": len(self._queue)}

    def close(self):
        with self._condition:
            self._closing = True
            self._condition.notify()
        self._thread.join()
