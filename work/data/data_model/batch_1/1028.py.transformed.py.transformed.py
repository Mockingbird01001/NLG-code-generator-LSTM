
from __future__ import unicode_literals
from . import Infinite
class Spinner(Infinite):
    phases = ('-', '\\', '|', '/')
    hide_cursor = True
    def update(self):
        i = self.index % len(self.phases)
        self.write(self.phases[i])
class PieSpinner(Spinner):
    phases = ['◷', '◶', '◵', '◴']
class MoonSpinner(Spinner):
    phases = ['◑', '◒', '◐', '◓']
class LineSpinner(Spinner):
    phases = ['⎺', '⎻', '⎼', '⎽', '⎼', '⎻']
class PixelSpinner(Spinner):
    phases = ['⣾', '⣷', '⣯', '⣟', '⡿', '⢿', '⣻', '⣽']
