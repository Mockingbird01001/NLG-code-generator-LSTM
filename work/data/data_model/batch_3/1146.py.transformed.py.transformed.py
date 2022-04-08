
import numpy as np
def func(a: str, b: int) -> None: ...
class Write:
    def write(self, value: str) -> None: ...
reveal_type(np.seterr(all=None))
reveal_type(np.seterr(divide="ignore"))
reveal_type(np.seterr(over="warn"))
reveal_type(np.seterr(under="call"))
reveal_type(np.seterr(invalid="raise"))
reveal_type(np.geterr())
reveal_type(np.setbufsize(4096))
reveal_type(np.getbufsize())
reveal_type(np.seterrcall(func))
reveal_type(np.seterrcall(Write()))
reveal_type(np.geterrcall())
reveal_type(np.errstate(call=func, all="call"))
reveal_type(np.errstate(call=Write(), divide="log", over="log"))
