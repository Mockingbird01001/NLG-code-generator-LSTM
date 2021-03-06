
from __future__ import unicode_literals
from . import Infinite, Progress
class Counter(Infinite):
    def update(self):
        self.write(str(self.index))
class Countdown(Progress):
    def update(self):
        self.write(str(self.remaining))
class Stack(Progress):
    phases = (' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█')
    def update(self):
        nphases = len(self.phases)
        i = min(nphases - 1, int(self.progress * nphases))
        self.write(self.phases[i])
class Pie(Stack):
    phases = ('○', '◔', '◑', '◕', '●')
