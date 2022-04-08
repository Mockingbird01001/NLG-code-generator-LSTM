from __future__ import absolute_import
from __future__ import division
import itertools
import sys
from signal import signal, SIGINT, default_int_handler
import time
import contextlib
import logging
from pip.compat import WINDOWS
from pip.utils import format_size
from pip.utils.logging import get_indentation
from pip._vendor import six
from pip._vendor.progress.bar import Bar, IncrementalBar
from pip._vendor.progress.helpers import (WritelnMixin,
                                          HIDE_CURSOR, SHOW_CURSOR)
from pip._vendor.progress.spinner import Spinner
try:
    from pip._vendor import colorama
except Exception:
    colorama = None
logger = logging.getLogger(__name__)
def _select_progress_class(preferred, fallback):
    encoding = getattr(preferred.file, "encoding", None)
    if not encoding:
        return fallback
    characters = [
        getattr(preferred, "empty_fill", six.text_type()),
        getattr(preferred, "fill", six.text_type()),
    ]
    characters += list(getattr(preferred, "phases", []))
    try:
        six.text_type().join(characters).encode(encoding)
    except UnicodeEncodeError:
        return fallback
    else:
        return preferred
_BaseBar = _select_progress_class(IncrementalBar, Bar)
class InterruptibleMixin(object):
    def __init__(self, *args, **kwargs):
        super(InterruptibleMixin, self).__init__(*args, **kwargs)
        self.original_handler = signal(SIGINT, self.handle_sigint)
        if self.original_handler is None:
            self.original_handler = default_int_handler
    def finish(self):
        super(InterruptibleMixin, self).finish()
        signal(SIGINT, self.original_handler)
    def handle_sigint(self, signum, frame):
        self.finish()
        self.original_handler(signum, frame)
class DownloadProgressMixin(object):
    def __init__(self, *args, **kwargs):
        super(DownloadProgressMixin, self).__init__(*args, **kwargs)
        self.message = (" " * (get_indentation() + 2)) + self.message
    @property
    def downloaded(self):
        return format_size(self.index)
    @property
    def download_speed(self):
        if self.avg == 0.0:
            return "..."
        return format_size(1 / self.avg) + "/s"
    @property
    def pretty_eta(self):
        if self.eta:
            return "eta %s" % self.eta_td
        return ""
    def iter(self, it, n=1):
        for x in it:
            yield x
            self.next(n)
        self.finish()
class WindowsMixin(object):
    def __init__(self, *args, **kwargs):
        if WINDOWS and self.hide_cursor:
            self.hide_cursor = False
        super(WindowsMixin, self).__init__(*args, **kwargs)
        if WINDOWS and colorama:
            self.file = colorama.AnsiToWin32(self.file)
            self.file.isatty = lambda: self.file.wrapped.isatty()
            self.file.flush = lambda: self.file.wrapped.flush()
class DownloadProgressBar(WindowsMixin, InterruptibleMixin,
                          DownloadProgressMixin, _BaseBar):
    file = sys.stdout
    message = "%(percent)d%%"
    suffix = "%(downloaded)s %(download_speed)s %(pretty_eta)s"
class DownloadProgressSpinner(WindowsMixin, InterruptibleMixin,
                              DownloadProgressMixin, WritelnMixin, Spinner):
    file = sys.stdout
    suffix = "%(downloaded)s %(download_speed)s"
    def next_phase(self):
        if not hasattr(self, "_phaser"):
            self._phaser = itertools.cycle(self.phases)
        return next(self._phaser)
    def update(self):
        message = self.message % self
        phase = self.next_phase()
        suffix = self.suffix % self
        line = ''.join([
            message,
            " " if message else "",
            phase,
            " " if suffix else "",
            suffix,
        ])
        self.writeln(line)
@contextlib.contextmanager
def hidden_cursor(file):
    if WINDOWS:
        yield
    elif not file.isatty() or logger.getEffectiveLevel() > logging.INFO:
        yield
    else:
        file.write(HIDE_CURSOR)
        try:
            yield
        finally:
            file.write(SHOW_CURSOR)
class RateLimiter(object):
    def __init__(self, min_update_interval_seconds):
        self._min_update_interval_seconds = min_update_interval_seconds
        self._last_update = 0
    def ready(self):
        now = time.time()
        delta = now - self._last_update
        return delta >= self._min_update_interval_seconds
    def reset(self):
        self._last_update = time.time()
class InteractiveSpinner(object):
    def __init__(self, message, file=None, spin_chars="-\\|/",
                 min_update_interval_seconds=0.125):
        self._message = message
        if file is None:
            file = sys.stdout
        self._file = file
        self._rate_limiter = RateLimiter(min_update_interval_seconds)
        self._finished = False
        self._spin_cycle = itertools.cycle(spin_chars)
        self._file.write(" " * get_indentation() + self._message + " ... ")
        self._width = 0
    def _write(self, status):
        assert not self._finished
        backup = "\b" * self._width
        self._file.write(backup + " " * self._width + backup)
        self._file.write(status)
        self._width = len(status)
        self._file.flush()
        self._rate_limiter.reset()
    def spin(self):
        if self._finished:
            return
        if not self._rate_limiter.ready():
            return
        self._write(next(self._spin_cycle))
    def finish(self, final_status):
        if self._finished:
            return
        self._write(final_status)
        self._file.write("\n")
        self._file.flush()
        self._finished = True
class NonInteractiveSpinner(object):
    def __init__(self, message, min_update_interval_seconds=60):
        self._message = message
        self._finished = False
        self._rate_limiter = RateLimiter(min_update_interval_seconds)
        self._update("started")
    def _update(self, status):
        assert not self._finished
        self._rate_limiter.reset()
        logger.info("%s: %s", self._message, status)
    def spin(self):
        if self._finished:
            return
        if not self._rate_limiter.ready():
            return
        self._update("still running...")
    def finish(self, final_status):
        if self._finished:
            return
        self._update("finished with status '%s'" % (final_status,))
        self._finished = True
@contextlib.contextmanager
def open_spinner(message):
    if sys.stdout.isatty() and logger.getEffectiveLevel() <= logging.INFO:
        spinner = InteractiveSpinner(message)
    else:
        spinner = NonInteractiveSpinner(message)
    try:
        with hidden_cursor(sys.stdout):
            yield spinner
    except KeyboardInterrupt:
        spinner.finish("canceled")
        raise
    except Exception:
        spinner.finish("error")
        raise
    else:
        spinner.finish("done")
