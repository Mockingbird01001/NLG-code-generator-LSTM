
__ALL__ = [
    'ModuleDeprecationWarning', 'VisibleDeprecationWarning', '_NoValue'
    ]
if '_is_loaded' in globals():
    raise RuntimeError('Reloading numpy._globals is not allowed')
_is_loaded = True
class ModuleDeprecationWarning(DeprecationWarning):
ModuleDeprecationWarning.__module__ = 'numpy'
class VisibleDeprecationWarning(UserWarning):
VisibleDeprecationWarning.__module__ = 'numpy'
class _NoValueType:
    __instance = None
    def __new__(cls):
        if not cls.__instance:
            cls.__instance = super(_NoValueType, cls).__new__(cls)
        return cls.__instance
    def __reduce__(self):
        return (self.__class__, ())
    def __repr__(self):
        return "<no value>"
_NoValue = _NoValueType()
