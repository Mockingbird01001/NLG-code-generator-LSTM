import numpy as np
a: "np.flatiter[np.ndarray]"
reveal_type(a.base)
reveal_type(a.copy())
reveal_type(a.coords)
reveal_type(a.index)
reveal_type(iter(a))
reveal_type(next(a))
reveal_type(a[0])
reveal_type(a[[0, 1, 2]])
reveal_type(a[...])
reveal_type(a[:])
