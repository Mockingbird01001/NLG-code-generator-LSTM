
import os
import numba as nb
import numpy as np
from cffi import FFI
from numpy.random import PCG64
ffi = FFI()
if os.path.exists('./distributions.dll'):
    lib = ffi.dlopen('./distributions.dll')
elif os.path.exists('./libdistributions.so'):
    lib = ffi.dlopen('./libdistributions.so')
else:
    raise RuntimeError('Required DLL/so file was not found.')
ffi.cdef("""
double random_standard_normal(void *bitgen_state);
""")
x = PCG64()
xffi = x.cffi
bit_generator = xffi.bit_generator
random_standard_normal = lib.random_standard_normal
def normals(n, bit_generator):
    out = np.empty(n)
    for i in range(n):
        out[i] = random_standard_normal(bit_generator)
    return out
normalsj = nb.jit(normals, nopython=True)
bit_generator_address = int(ffi.cast('uintptr_t', bit_generator))
norm = normalsj(1000, bit_generator_address)
print(norm[:12])
