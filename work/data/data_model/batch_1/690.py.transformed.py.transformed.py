
from numpy.core import umath as um
__all__ = ['NDArrayOperatorsMixin']
def _disables_array_ufunc(obj):
    try:
        return obj.__array_ufunc__ is None
    except AttributeError:
        return False
def _binary_method(ufunc, name):
    def func(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return ufunc(self, other)
    func.__name__ = '__{}__'.format(name)
    return func
def _reflected_binary_method(ufunc, name):
    def func(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return ufunc(other, self)
    func.__name__ = '__r{}__'.format(name)
    return func
def _inplace_binary_method(ufunc, name):
    def func(self, other):
        return ufunc(self, other, out=(self,))
    func.__name__ = '__i{}__'.format(name)
    return func
def _numeric_methods(ufunc, name):
    return (_binary_method(ufunc, name),
            _reflected_binary_method(ufunc, name),
            _inplace_binary_method(ufunc, name))
def _unary_method(ufunc, name):
    def func(self):
        return ufunc(self)
    func.__name__ = '__{}__'.format(name)
    return func
class NDArrayOperatorsMixin:
    __lt__ = _binary_method(um.less, 'lt')
    __le__ = _binary_method(um.less_equal, 'le')
    __eq__ = _binary_method(um.equal, 'eq')
    __ne__ = _binary_method(um.not_equal, 'ne')
    __gt__ = _binary_method(um.greater, 'gt')
    __ge__ = _binary_method(um.greater_equal, 'ge')
    __add__, __radd__, __iadd__ = _numeric_methods(um.add, 'add')
    __sub__, __rsub__, __isub__ = _numeric_methods(um.subtract, 'sub')
    __mul__, __rmul__, __imul__ = _numeric_methods(um.multiply, 'mul')
    __matmul__, __rmatmul__, __imatmul__ = _numeric_methods(
        um.matmul, 'matmul')
    __truediv__, __rtruediv__, __itruediv__ = _numeric_methods(
        um.true_divide, 'truediv')
    __floordiv__, __rfloordiv__, __ifloordiv__ = _numeric_methods(
        um.floor_divide, 'floordiv')
    __mod__, __rmod__, __imod__ = _numeric_methods(um.remainder, 'mod')
    __divmod__ = _binary_method(um.divmod, 'divmod')
    __rdivmod__ = _reflected_binary_method(um.divmod, 'divmod')
    __pow__, __rpow__, __ipow__ = _numeric_methods(um.power, 'pow')
    __lshift__, __rlshift__, __ilshift__ = _numeric_methods(
        um.left_shift, 'lshift')
    __rshift__, __rrshift__, __irshift__ = _numeric_methods(
        um.right_shift, 'rshift')
    __and__, __rand__, __iand__ = _numeric_methods(um.bitwise_and, 'and')
    __xor__, __rxor__, __ixor__ = _numeric_methods(um.bitwise_xor, 'xor')
    __or__, __ror__, __ior__ = _numeric_methods(um.bitwise_or, 'or')
    __neg__ = _unary_method(um.negative, 'neg')
    __pos__ = _unary_method(um.positive, 'pos')
    __abs__ = _unary_method(um.absolute, 'abs')
    __invert__ = _unary_method(um.invert, 'invert')
