
class InputState(object):
    PURE_ASCII = 0
    ESC_ASCII = 1
    HIGH_BYTE = 2
class LanguageFilter(object):
    CHINESE_SIMPLIFIED = 0x01
    CHINESE_TRADITIONAL = 0x02
    JAPANESE = 0x04
    KOREAN = 0x08
    NON_CJK = 0x10
    ALL = 0x1F
    CHINESE = CHINESE_SIMPLIFIED | CHINESE_TRADITIONAL
    CJK = CHINESE | JAPANESE | KOREAN
class ProbingState(object):
    DETECTING = 0
    FOUND_IT = 1
    NOT_ME = 2
class MachineState(object):
    START = 0
    ERROR = 1
    ITS_ME = 2
class SequenceLikelihood(object):
    NEGATIVE = 0
    UNLIKELY = 1
    LIKELY = 2
    POSITIVE = 3
    @classmethod
    def get_num_categories(cls):
        return 4
class CharacterCategory(object):
    UNDEFINED = 255
    LINE_BREAK = 254
    SYMBOL = 253
    DIGIT = 252
    CONTROL = 251
