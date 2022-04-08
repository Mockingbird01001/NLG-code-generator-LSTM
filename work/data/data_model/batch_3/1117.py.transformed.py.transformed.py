
import sys
from pip._vendor.tenacity import BaseRetrying
from pip._vendor.tenacity import DoAttempt
from pip._vendor.tenacity import DoSleep
from pip._vendor.tenacity import RetryCallState
from tornado import gen
class TornadoRetrying(BaseRetrying):
    def __init__(self, sleep=gen.sleep, **kwargs):
        super(TornadoRetrying, self).__init__(**kwargs)
        self.sleep = sleep
    @gen.coroutine
    def __call__(self, fn, *args, **kwargs):
        self.begin(fn)
        retry_state = RetryCallState(retry_object=self, fn=fn, args=args, kwargs=kwargs)
        while True:
            do = self.iter(retry_state=retry_state)
            if isinstance(do, DoAttempt):
                try:
                    result = yield fn(*args, **kwargs)
                except BaseException:
                    retry_state.set_exception(sys.exc_info())
                else:
                    retry_state.set_result(result)
            elif isinstance(do, DoSleep):
                retry_state.prepare_for_next_attempt()
                yield self.sleep(do)
            else:
                raise gen.Return(do)
