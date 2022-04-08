import errno
import itertools
import logging
import os.path
import tempfile
from contextlib import ExitStack, contextmanager
from typing import Any, Dict, Iterator, Optional, TypeVar, Union
from pip._internal.utils.misc import enum, rmtree
logger = logging.getLogger(__name__)
_T = TypeVar("_T", bound="TempDirectory")
tempdir_kinds = enum(
    BUILD_ENV="build-env",
    EPHEM_WHEEL_CACHE="ephem-wheel-cache",
    REQ_BUILD="req-build",
)
_tempdir_manager = None
@contextmanager
def global_tempdir_manager():
    global _tempdir_manager
    with ExitStack() as stack:
        old_tempdir_manager, _tempdir_manager = _tempdir_manager, stack
        try:
            yield
        finally:
            _tempdir_manager = old_tempdir_manager
class TempDirectoryTypeRegistry:
    def __init__(self):
        self._should_delete = {}
    def set_delete(self, kind, value):
        self._should_delete[kind] = value
    def get_delete(self, kind):
        return self._should_delete.get(kind, True)
_tempdir_registry = None
@contextmanager
def tempdir_registry():
    global _tempdir_registry
    old_tempdir_registry = _tempdir_registry
    _tempdir_registry = TempDirectoryTypeRegistry()
    try:
        yield _tempdir_registry
    finally:
        _tempdir_registry = old_tempdir_registry
class _Default:
    pass
_default = _Default()
class TempDirectory:
    def __init__(
        self,
        path=None,
        delete=_default,
        kind="temp",
        globally_managed=False,
    ):
        super().__init__()
        if delete is _default:
            if path is not None:
                delete = False
            else:
                delete = None
        if path is None:
            path = self._create(kind)
        self._path = path
        self._deleted = False
        self.delete = delete
        self.kind = kind
        if globally_managed:
            assert _tempdir_manager is not None
            _tempdir_manager.enter_context(self)
    @property
    def path(self):
        assert not self._deleted, f"Attempted to access deleted path: {self._path}"
        return self._path
    def __repr__(self):
        return f"<{self.__class__.__name__} {self.path!r}>"
    def __enter__(self):
        return self
    def __exit__(self, exc, value, tb):
        if self.delete is not None:
            delete = self.delete
        elif _tempdir_registry:
            delete = _tempdir_registry.get_delete(self.kind)
        else:
            delete = True
        if delete:
            self.cleanup()
    def _create(self, kind):
        path = os.path.realpath(tempfile.mkdtemp(prefix=f"pip-{kind}-"))
        logger.debug("Created temporary directory: %s", path)
        return path
    def cleanup(self):
        self._deleted = True
        if not os.path.exists(self._path):
            return
        rmtree(self._path)
class AdjacentTempDirectory(TempDirectory):
    LEADING_CHARS = "-~.=%0123456789"
    def __init__(self, original, delete=None):
        self.original = original.rstrip("/\\")
        super().__init__(delete=delete)
    @classmethod
    def _generate_names(cls, name):
        for i in range(1, len(name)):
            for candidate in itertools.combinations_with_replacement(
                cls.LEADING_CHARS, i - 1
            ):
                new_name = "~" + "".join(candidate) + name[i:]
                if new_name != name:
                    yield new_name
        for i in range(len(cls.LEADING_CHARS)):
            for candidate in itertools.combinations_with_replacement(
                cls.LEADING_CHARS, i
            ):
                new_name = "~" + "".join(candidate) + name
                if new_name != name:
                    yield new_name
    def _create(self, kind):
        root, name = os.path.split(self.original)
        for candidate in self._generate_names(name):
            path = os.path.join(root, candidate)
            try:
                os.mkdir(path)
            except OSError as ex:
                if ex.errno != errno.EEXIST:
                    raise
            else:
                path = os.path.realpath(path)
                break
        else:
            path = os.path.realpath(tempfile.mkdtemp(prefix=f"pip-{kind}-"))
        logger.debug("Created temporary directory: %s", path)
        return path
