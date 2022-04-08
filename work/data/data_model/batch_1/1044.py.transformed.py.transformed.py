
import abc
from pip._vendor import six
@six.add_metaclass(abc.ABCMeta)
class stop_base(object):
    @abc.abstractmethod
    def __call__(self, retry_state):
        pass
    def __and__(self, other):
        return stop_all(self, other)
    def __or__(self, other):
        return stop_any(self, other)
class stop_any(stop_base):
    def __init__(self, *stops):
        self.stops = stops
    def __call__(self, retry_state):
        return any(x(retry_state) for x in self.stops)
class stop_all(stop_base):
    def __init__(self, *stops):
        self.stops = stops
    def __call__(self, retry_state):
        return all(x(retry_state) for x in self.stops)
class _stop_never(stop_base):
    def __call__(self, retry_state):
        return False
stop_never = _stop_never()
class stop_when_event_set(stop_base):
    def __init__(self, event):
        self.event = event
    def __call__(self, retry_state):
        return self.event.is_set()
class stop_after_attempt(stop_base):
    def __init__(self, max_attempt_number):
        self.max_attempt_number = max_attempt_number
    def __call__(self, retry_state):
        return retry_state.attempt_number >= self.max_attempt_number
class stop_after_delay(stop_base):
    def __init__(self, max_delay):
        self.max_delay = max_delay
    def __call__(self, retry_state):
        return retry_state.seconds_since_start >= self.max_delay
