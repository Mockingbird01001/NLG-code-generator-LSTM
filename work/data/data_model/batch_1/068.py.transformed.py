
import logging
import re
from .enums import ProbingState
class CharSetProber(object):
    SHORTCUT_THRESHOLD = 0.95
    def __init__(self, lang_filter=None):
        self._state = None
        self.lang_filter = lang_filter
        self.logger = logging.getLogger(__name__)
    def reset(self):
        self._state = ProbingState.DETECTING
    @property
    def charset_name(self):
        return None
    def feed(self, buf):
        pass
    @property
    def state(self):
        return self._state
    def get_confidence(self):
        return 0.0
    @staticmethod
    def filter_high_byte_only(buf):
        buf = re.sub(b'([\x00-\x7F])+', b' ', buf)
        return buf
    @staticmethod
    def filter_international_words(buf):
        filtered = bytearray()
        words = re.findall(b'[a-zA-Z]*[\x80-\xFF]+[a-zA-Z]*[^a-zA-Z\x80-\xFF]?',
                           buf)
        for word in words:
            filtered.extend(word[:-1])
            last_char = word[-1:]
            if not last_char.isalpha() and last_char < b'\x80':
                last_char = b' '
            filtered.extend(last_char)
        return filtered
    @staticmethod
    def filter_with_english_letters(buf):
        filtered = bytearray()
        in_tag = False
        prev = 0
        for curr in range(len(buf)):
            buf_char = buf[curr:curr + 1]
            if buf_char == b'>':
                in_tag = False
            elif buf_char == b'<':
                in_tag = True
            if buf_char < b'\x80' and not buf_char.isalpha():
                if curr > prev and not in_tag:
                    filtered.extend(buf[prev:curr])
                    filtered.extend(b' ')
                prev = curr + 1
        if not in_tag:
            filtered.extend(buf[prev:])
        return filtered
