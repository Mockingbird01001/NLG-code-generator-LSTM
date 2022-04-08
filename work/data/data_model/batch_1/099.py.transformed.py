
import datetime
from collections import deque
from bson.py3compat import integer_types
from pymongo import helpers
from pymongo.errors import AutoReconnect, NotMasterError, OperationFailure
from pymongo.message import _CursorAddress, _GetMore, _convert_exception
class CommandCursor(object):
    def __init__(self, collection, cursor_info, address, retrieved=0):
        self.__collection = collection
        self.__id = cursor_info['id']
        self.__address = address
        self.__data = deque(cursor_info['firstBatch'])
        self.__retrieved = retrieved
        self.__batch_size = 0
        self.__killed = (self.__id == 0)
        if "ns" in cursor_info:
            self.__ns = cursor_info["ns"]
        else:
            self.__ns = collection.full_name
    def __del__(self):
        if self.__id and not self.__killed:
            self.__die()
    def __die(self, synchronous=False):
        if self.__id and not self.__killed:
            address = _CursorAddress(
                self.__address, self.__collection.full_name)
            if synchronous:
                self.__collection.database.client._close_cursor_now(
                    self.__id, address)
            else:
                self.__collection.database.client.close_cursor(
                    self.__id, address)
        self.__killed = True
    def close(self):
        self.__die(True)
    def batch_size(self, batch_size):
        if not isinstance(batch_size, integer_types):
            raise TypeError("batch_size must be an integer")
        if batch_size < 0:
            raise ValueError("batch_size must be >= 0")
        self.__batch_size = batch_size == 1 and 2 or batch_size
        return self
    def __send_message(self, operation):
        client = self.__collection.database.client
        listeners = client._event_listeners
        publish = listeners.enabled_for_commands
        try:
            response = client._send_message_with_response(
                operation, address=self.__address)
        except AutoReconnect:
            self.__killed = True
            raise
        cmd_duration = response.duration
        rqst_id = response.request_id
        from_command = response.from_command
        if publish:
            start = datetime.datetime.now()
        try:
            doc = helpers._unpack_response(response.data,
                                           self.__id,
                                           self.__collection.codec_options)
            if from_command:
                helpers._check_command_response(doc['data'][0])
        except OperationFailure as exc:
            self.__killed = True
            if publish:
                duration = (datetime.datetime.now() - start) + cmd_duration
                listeners.publish_command_failure(
                    duration, exc.details, "getMore", rqst_id, self.__address)
            raise
        except NotMasterError as exc:
            self.__killed = True
            if publish:
                duration = (datetime.datetime.now() - start) + cmd_duration
                listeners.publish_command_failure(
                    duration, exc.details, "getMore", rqst_id, self.__address)
            client._reset_server_and_request_check(self.address)
            raise
        except Exception as exc:
            if publish:
                duration = (datetime.datetime.now() - start) + cmd_duration
                listeners.publish_command_failure(
                    duration, _convert_exception(exc), "getMore", rqst_id,
                    self.__address)
            raise
        if from_command:
            cursor = doc['data'][0]['cursor']
            documents = cursor['nextBatch']
            self.__id = cursor['id']
            self.__retrieved += len(documents)
        else:
            documents = doc["data"]
            self.__id = doc["cursor_id"]
            self.__retrieved += doc["number_returned"]
        if publish:
            duration = (datetime.datetime.now() - start) + cmd_duration
            res = {"cursor": {"id": self.__id,
                              "ns": self.__collection.full_name,
                              "nextBatch": documents},
                   "ok": 1}
            listeners.publish_command_success(
                duration, res, "getMore", rqst_id, self.__address)
        if self.__id == 0:
            self.__killed = True
        self.__data = deque(documents)
    def _refresh(self):
        if len(self.__data) or self.__killed:
            return len(self.__data)
        if self.__id:
            dbname, collname = self.__ns.split('.', 1)
            self.__send_message(
                _GetMore(dbname,
                         collname,
                         self.__batch_size,
                         self.__id,
                         self.__collection.codec_options))
        else:
            self.__killed = True
        return len(self.__data)
    @property
    def alive(self):
        return bool(len(self.__data) or (not self.__killed))
    @property
    def cursor_id(self):
        return self.__id
    @property
    def address(self):
        return self.__address
    def __iter__(self):
        return self
    def next(self):
        if len(self.__data) or self._refresh():
            coll = self.__collection
            return coll.database._fix_outgoing(self.__data.popleft(), coll)
        else:
            raise StopIteration
    __next__ = next
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
