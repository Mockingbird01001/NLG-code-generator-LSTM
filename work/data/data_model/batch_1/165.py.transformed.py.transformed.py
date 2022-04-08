
import logging
from .enums import MachineState
class CodingStateMachine(object):
    def __init__(self, sm):
        self._model = sm
        self._curr_byte_pos = 0
        self._curr_char_len = 0
        self._curr_state = None
        self.logger = logging.getLogger(__name__)
        self.reset()
    def reset(self):
        self._curr_state = MachineState.START
    def next_state(self, c):
        byte_class = self._model['class_table'][c]
        if self._curr_state == MachineState.START:
            self._curr_byte_pos = 0
            self._curr_char_len = self._model['char_len_table'][byte_class]
        curr_state = (self._curr_state * self._model['class_factor']
                      + byte_class)
        self._curr_state = self._model['state_table'][curr_state]
        self._curr_byte_pos += 1
        return self._curr_state
    def get_current_charlen(self):
        return self._curr_char_len
    def get_coding_state_machine(self):
        return self._model['name']
    @property
    def language(self):
        return self._model['language']
