
import time
def sleep(seconds):
    time.sleep(seconds)
class sleep_using_event(object):
    def __init__(self, event):
        self.event = event
    def __call__(self, timeout):
        self.event.wait(timeout=timeout)
