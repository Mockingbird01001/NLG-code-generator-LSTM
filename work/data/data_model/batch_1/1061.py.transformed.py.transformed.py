from __future__ import absolute_import
import os
import time
from . import (LockBase, NotLocked, NotMyLock, LockTimeout,
               AlreadyLocked)
class SymlinkLockFile(LockBase):
    def __init__(self, path, threaded=True, timeout=None):
        LockBase.__init__(self, path, threaded, timeout)
        self.unique_name = os.path.split(self.unique_name)[1]
    def acquire(self, timeout=None):
        timeout = timeout if timeout is not None else self.timeout
        end_time = time.time()
        if timeout is not None and timeout > 0:
            end_time += timeout
        while True:
            try:
                os.symlink(self.unique_name, self.lock_file)
            except OSError:
                if self.i_am_locking():
                    return
                else:
                    if timeout is not None and time.time() > end_time:
                        if timeout > 0:
                            raise LockTimeout("Timeout waiting to acquire"
                                              " lock for %s" %
                                              self.path)
                        else:
                            raise AlreadyLocked("%s is already locked" %
                                                self.path)
                    time.sleep(timeout / 10 if timeout is not None else 0.1)
            else:
                return
    def release(self):
        if not self.is_locked():
            raise NotLocked("%s is not locked" % self.path)
        elif not self.i_am_locking():
            raise NotMyLock("%s is locked, but not by me" % self.path)
        os.unlink(self.lock_file)
    def is_locked(self):
        return os.path.islink(self.lock_file)
    def i_am_locking(self):
        return (os.path.islink(self.lock_file)
                and os.readlink(self.lock_file) == self.unique_name)
    def break_lock(self):
        if os.path.islink(self.lock_file):
            os.unlink(self.lock_file)
