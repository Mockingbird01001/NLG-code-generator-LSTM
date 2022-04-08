import numpy as np
reveal_type(np.issctype(np.generic))
reveal_type(np.issctype("foo"))
reveal_type(np.obj2sctype("S8"))
reveal_type(np.obj2sctype("S8", default=None))
reveal_type(
    np.obj2sctype("foo", default=int)
)
reveal_type(np.issubclass_(np.float64, float))
reveal_type(np.issubclass_(np.float64, (int, float)))
reveal_type(np.sctype2char("S8"))
reveal_type(np.sctype2char(list))
reveal_type(np.find_common_type([np.int64], [np.int64]))
