
from .core import Input
__all__ = (
    'ColorInput', 'DateInput', 'DateTimeInput', 'DateTimeLocalInput',
    'EmailInput', 'MonthInput', 'NumberInput', 'RangeInput', 'SearchInput',
    'TelInput', 'TimeInput', 'URLInput', 'WeekInput',
)
class SearchInput(Input):
    input_type = 'search'
class TelInput(Input):
    input_type = 'tel'
class URLInput(Input):
    input_type = 'url'
class EmailInput(Input):
    input_type = 'email'
class DateTimeInput(Input):
    input_type = 'datetime'
class DateInput(Input):
    input_type = 'date'
class MonthInput(Input):
    input_type = 'month'
class WeekInput(Input):
    input_type = 'week'
class TimeInput(Input):
    input_type = 'time'
class DateTimeLocalInput(Input):
    input_type = 'datetime-local'
class NumberInput(Input):
    input_type = 'number'
    def __init__(self, step=None, min=None, max=None):
        self.step = step
        self.min = min
        self.max = max
    def __call__(self, field, **kwargs):
        if self.step is not None:
            kwargs.setdefault('step', self.step)
        if self.min is not None:
            kwargs.setdefault('min', self.min)
        if self.max is not None:
            kwargs.setdefault('max', self.max)
        return super(NumberInput, self).__call__(field, **kwargs)
class RangeInput(Input):
    input_type = 'range'
    def __init__(self, step=None):
        self.step = step
    def __call__(self, field, **kwargs):
        if self.step is not None:
            kwargs.setdefault('step', self.step)
        return super(RangeInput, self).__call__(field, **kwargs)
class ColorInput(Input):
    input_type = 'color'
