import itertools
import sys
from signal import SIGINT, default_int_handler, signal
from typing import Any, Dict, List
from pip._vendor.progress.bar import Bar, FillingCirclesBar, IncrementalBar
from pip._vendor.progress.spinner import Spinner
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.logging import get_indentation
from pip._internal.utils.misc import format_size
try:
    from pip._vendor import colorama
except Exception:
    colorama = None
def _select_progress_class(preferred, fallback):
    encoding = getattr(preferred.file, "encoding", None)
    if not encoding:
        return fallback
    characters = [
        getattr(preferred, "empty_fill", ""),
        getattr(preferred, "fill", ""),
    ]
    characters += list(getattr(preferred, "phases", []))
    try:
        .join(characters).encode(encoding)
    except UnicodeEncodeError:
        return fallback
    else:
        return preferred
_BaseBar = _select_progress_class(IncrementalBar, Bar)
class InterruptibleMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_handler = signal(SIGINT, self.handle_sigint)
        if self.original_handler is None:
            self.original_handler = default_int_handler
    def finish(self):
        super().finish()
        signal(SIGINT, self.original_handler)
    def handle_sigint(self, signum, frame):
        self.finish()
        self.original_handler(signum, frame)
class SilentBar(Bar):
    def update(self):
        pass
class BlueEmojiBar(IncrementalBar):
    suffix = "%(percent)d%%"
    bar_prefix = " "
    bar_suffix = " "
    phases = ("\U0001F539", "\U0001F537", "\U0001F535")
class DownloadProgressMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
            return f"eta {self.eta_td}"
        return ""
    def iter(self, it):
        for x in it:
            yield x
            self.next(len(x))
        self.finish()
class WindowsMixin:
    def __init__(self, *args, **kwargs):
        if WINDOWS and self.hide_cursor:
            self.hide_cursor = False
        super().__init__(*args, **kwargs)
        if WINDOWS and colorama:
            self.file = colorama.AnsiToWin32(self.file)
            self.file.isatty = lambda: self.file.wrapped.isatty()
            self.file.flush = lambda: self.file.wrapped.flush()
class BaseDownloadProgressBar(WindowsMixin, InterruptibleMixin, DownloadProgressMixin):
    file = sys.stdout
    message = "%(percent)d%%"
    suffix = "%(downloaded)s %(download_speed)s %(pretty_eta)s"
class DefaultDownloadProgressBar(BaseDownloadProgressBar, _BaseBar):
    pass
class DownloadSilentBar(BaseDownloadProgressBar, SilentBar):
    pass
class DownloadBar(BaseDownloadProgressBar, Bar):
    pass
class DownloadFillingCirclesBar(BaseDownloadProgressBar, FillingCirclesBar):
    pass
class DownloadBlueEmojiProgressBar(BaseDownloadProgressBar, BlueEmojiBar):
    pass
class DownloadProgressSpinner(
    WindowsMixin, InterruptibleMixin, DownloadProgressMixin, Spinner
):
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
        line = "".join(
            [
                message,
                " " if message else "",
                phase,
                " " if suffix else "",
                suffix,
            ]
        )
        self.writeln(line)
BAR_TYPES = {
    "off": (DownloadSilentBar, DownloadSilentBar),
    "on": (DefaultDownloadProgressBar, DownloadProgressSpinner),
    "ascii": (DownloadBar, DownloadProgressSpinner),
    "pretty": (DownloadFillingCirclesBar, DownloadProgressSpinner),
    "emoji": (DownloadBlueEmojiProgressBar, DownloadProgressSpinner),
}
def DownloadProgressProvider(progress_bar, max=None):
    if max is None or max == 0:
        return BAR_TYPES[progress_bar][1]().iter
    else:
        return BAR_TYPES[progress_bar][0](max=max).iter
