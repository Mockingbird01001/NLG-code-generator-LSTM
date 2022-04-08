
_builtin_next = next
_SENTINEL = object()
def newnext(iterator, default=_SENTINEL):
    try:
        try:
            return iterator.__next__()
        except AttributeError:
            try:
                return iterator.next()
            except AttributeError:
                raise TypeError("'{0}' object is not an iterator".format(
                                           iterator.__class__.__name__))
    except StopIteration as e:
        if default is _SENTINEL:
            raise e
        else:
            return default
__all__ = ['newnext']
