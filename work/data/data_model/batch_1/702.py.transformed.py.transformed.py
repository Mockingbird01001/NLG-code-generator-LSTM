
__all__ = ['time']
try:
    import monotime
except ImportError:
    pass
try:
    from monotonic import monotonic as time
except ImportError:
    try:
        from time import monotonic as time
    except ImportError:
        from time import time
