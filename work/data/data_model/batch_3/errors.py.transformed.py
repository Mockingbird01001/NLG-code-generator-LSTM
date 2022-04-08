
class PyCTError(Exception):
class UnsupportedLanguageElementError(PyCTError, NotImplementedError):
class InaccessibleSourceCodeError(PyCTError, ValueError):
