from io import BytesIO
class CallbackFileWrapper(object):
    def __init__(self, fp, callback):
        self.__buf = BytesIO()
        self.__fp = fp
        self.__callback = callback
    def __getattr__(self, name):
        fp = self.__getattribute__("_CallbackFileWrapper__fp")
        return getattr(fp, name)
    def __is_fp_closed(self):
        try:
            return self.__fp.fp is None
        except AttributeError:
            pass
        try:
            return self.__fp.closed
        except AttributeError:
            pass
        return False
    def _close(self):
        if self.__callback:
            self.__callback(self.__buf.getvalue())
        self.__callback = None
    def read(self, amt=None):
        data = self.__fp.read(amt)
        self.__buf.write(data)
        if self.__is_fp_closed():
            self._close()
        return data
    def _safe_read(self, amt):
        data = self.__fp._safe_read(amt)
        if amt == 2 and data == b"\r\n":
            return data
        self.__buf.write(data)
        if self.__is_fp_closed():
            self._close()
        return data
