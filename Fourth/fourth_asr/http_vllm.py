"""Continuous vLLM scheduling for the synchronous Fourth HTTP protocol."""
from itertools import count

from .batch_backend import BatchBackend


class HttpVLLMWorker(BatchBackend):
    scheduling_mode = "continuous"

    def _serve(self, backend):
        engine = backend.llm.llm_engine
        engine_dead_error = getattr(backend, "engine_dead_error", ())
        ids = count()
        pending = {}
        batch = []
        try:
            while True:
                batch = self._next_batch(len(pending))
                if not batch and not pending:
                    return
                before = len(pending)
                for _, call, future in batch:
                    try:
                        prompt, decode = backend.prepare_decode(*call)
                    except Exception as exc:
                        future.set_exception(exc)
                        continue
                    request_id = str(next(ids))
                    pending[request_id] = (future, decode)
                    try:
                        engine.add_request(request_id, prompt, backend.sampling)
                    except engine_dead_error:
                        raise
                    except Exception as exc:
                        # add_request can register a detokenizer before submitting to the core.
                        engine.abort_request([request_id])
                        del pending[request_id]
                        future.set_exception(exc)
                if len(pending) > before:
                    self._record_batch(len(pending))
                if pending:
                    self._step(engine, pending)
        except BaseException as exc:
            # A fatal engine/abort failure must also wake jobs already taken off the FIFO.
            with self._condition:
                self._closing = True
            for _, _, future in batch:
                if not future.done():
                    future.set_exception(exc)
            for future, _ in pending.values():
                if not future.done():
                    future.set_exception(exc)
            raise

    @staticmethod
    def _step(engine, pending):
        # The v0.11.1 output socket thread can exit after forwarding one plain
        # exception. A later step may block forever even if abort_request returned.
        for output in engine.step():
            if not output.finished:
                continue
            future, decode = pending[output.request_id]
            try:
                reply = decode(output)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(reply)
            del pending[output.request_id]
        # This engine is dedicated to HTTP with one completion per request. Never
        # wait forever for a final reply the engine has already dropped.
        if engine.get_num_unfinished_requests() != len(pending):
            raise RuntimeError("vLLM unfinished request count disagrees with pending HTTP replies")
