import numpy as np
np.sin(1)
np.sin([1, 2, 3])
np.sin(1, out=np.empty(1))
np.matmul(np.ones((2, 2, 2)), np.ones((2, 2, 2)), axes=[(0, 1), (0, 1), (0, 1)])
np.sin(1, signature="D")
np.sin(1, extobj=[16, 1, lambda: None])
np.sin.types[0]
np.sin.__name__
np.abs(np.array([1]))
