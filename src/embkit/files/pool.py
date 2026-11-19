import threading
from collections import OrderedDict
from contextvars import ContextVar
import os
import operator


_active_open_file_pool = ContextVar("embkit_open_file_pool", default=None)


class OpenFilePool:
    """
    Limit the files kept open by ``LargeCsvReader`` instances in this context.

    The pool is process-local and uses least-recently-used eviction. Readers
    join it automatically when accessed inside the context:

        with OpenFilePool(max_open=32):
            readers = [LargeCsvReader(path) for path in paths]
            rows = [reader[0] for reader in readers]

    On exit, every pooled file is closed. With ``DataLoader``, each worker
    maintains its own pool, so the system-wide limit is multiplied by the
    number of worker processes.
    """

    def __init__(self, max_open=32):
        try:
            max_open = operator.index(max_open)
        except TypeError:
            raise TypeError("max_open must be an integer") from None
        if max_open <= 0:
            raise ValueError("max_open must be greater than zero")

        self.max_open = max_open
        self._entries = OrderedDict()
        self._lock = threading.RLock()
        self._pid = os.getpid()
        self._active = False
        self._token = None

    @classmethod
    def current(cls):
        pool = _active_open_file_pool.get()
        return pool if pool is not None and pool._active else None

    @property
    def active(self):
        return self._active

    @property
    def open_count(self):
        with self._lock:
            self._reset_after_fork()
            return len(self._entries)

    def __enter__(self):
        if self._token is not None:
            raise RuntimeError("OpenFilePool cannot be entered more than once")
        self._active = True
        self._token = _active_open_file_pool.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        token = self._token
        self._token = None
        try:
            self.close()
        finally:
            self._active = False
            if token is not None:
                _active_open_file_pool.reset(token)

    def _reset_after_fork(self):
        pid = os.getpid()
        if pid == self._pid:
            return
        self._close_entries()
        self._pid = pid

    def _close_entries(self):
        while self._entries:
            _, (reader, handle) = self._entries.popitem(last=False)
            self._close_entry(reader, handle)

    @staticmethod
    def _close_entry(reader, handle):
        try:
            handle.close()
        finally:
            reader._release_pool_file(handle)

    @staticmethod
    def _is_closed(handle):
        closed = getattr(handle, "closed", None)
        if closed is not None:
            return closed

        wrapped_handle = getattr(handle, "_handle", None)
        if wrapped_handle is not None:
            closed = getattr(wrapped_handle, "closed", None)
            if closed is not None:
                return closed

        fileno = getattr(handle, "fileno", None)
        if fileno is None:
            return False
        try:
            fileno()
        except (OSError, ValueError):
            return True
        return False

    def acquire(self, reader):
        """Return the reader's handle, opening or reusing an entry as needed."""
        with self._lock:
            self._reset_after_fork()
            key = id(reader)
            entry = self._entries.get(key)
            if entry is not None and entry[0] is reader:
                handle = entry[1]
                if not self._is_closed(handle):
                    self._entries.move_to_end(key)
                    return handle
                self._entries.pop(key)
                reader._release_pool_file(handle)

            while len(self._entries) >= self.max_open:
                _, (evicted_reader, handle) = self._entries.popitem(last=False)
                self._close_entry(evicted_reader, handle)

            handle = reader._open()
            reader._adopt_pool_file(self, handle)
            self._entries[key] = (reader, handle)
            return handle

    def release(self, reader):
        """Close and remove a reader's handle, if the pool owns one."""
        with self._lock:
            self._reset_after_fork()
            entry = self._entries.pop(id(reader), None)
            if entry is not None and entry[0] is reader:
                self._close_entry(reader, entry[1])

    def close(self):
        """Close all files currently owned by this pool."""
        with self._lock:
            self._close_entries()
            self._pid = os.getpid()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_entries"] = OrderedDict()
        state["_lock"] = None
        state["_pid"] = os.getpid()
        state["_token"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._entries = OrderedDict()
        self._lock = threading.RLock()
        self._pid = os.getpid()
        self._token = None

