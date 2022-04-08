
import copy
import datetime
import warnings
from collections import deque
from bson import RE_TYPE
from bson.code import Code
from bson.py3compat import (iteritems,
                            integer_types,
                            string_type)
from bson.son import SON
from pymongo import helpers
from pymongo.common import validate_boolean, validate_is_mapping
from pymongo.collation import validate_collation_or_none
from pymongo.errors import (AutoReconnect,
                            ConnectionFailure,
                            InvalidOperation,
                            NotMasterError,
                            OperationFailure)
from pymongo.message import _CursorAddress, _GetMore, _Query, _convert_exception
from pymongo.read_preferences import ReadPreference
_QUERY_OPTIONS = {
    "tailable_cursor": 2,
    "slave_okay": 4,
    "oplog_replay": 8,
    "no_timeout": 16,
    "await_data": 32,
    "exhaust": 64,
    "partial": 128}
class CursorType(object):
    NON_TAILABLE = 0
    TAILABLE = _QUERY_OPTIONS["tailable_cursor"]
    TAILABLE_AWAIT = TAILABLE | _QUERY_OPTIONS["await_data"]
    EXHAUST = _QUERY_OPTIONS["exhaust"]
class _SocketManager:
    def __init__(self, sock, pool):
        self.sock = sock
        self.pool = pool
        self.__closed = False
    def __del__(self):
        self.close()
    def close(self):
        if not self.__closed:
            self.__closed = True
            self.pool.return_socket(self.sock)
            self.sock, self.pool = None, None
class Cursor(object):
    def __init__(self, collection, filter=None, projection=None, skip=0,
                 limit=0, no_cursor_timeout=False,
                 cursor_type=CursorType.NON_TAILABLE,
                 sort=None, allow_partial_results=False, oplog_replay=False,
                 modifiers=None, batch_size=0, manipulate=True,
                 collation=None, hint=None, max_scan=None, max_time_ms=None,
                 max=None, min=None, return_key=False, show_record_id=False,
                 snapshot=False, comment=None):
        self.__id = None
        self.__exhaust = False
        self.__exhaust_mgr = None
        spec = filter
        if spec is None:
            spec = {}
        validate_is_mapping("filter", spec)
        if not isinstance(skip, int):
            raise TypeError("skip must be an instance of int")
        if not isinstance(limit, int):
            raise TypeError("limit must be an instance of int")
        validate_boolean("no_cursor_timeout", no_cursor_timeout)
        if cursor_type not in (CursorType.NON_TAILABLE, CursorType.TAILABLE,
                               CursorType.TAILABLE_AWAIT, CursorType.EXHAUST):
            raise ValueError("not a valid value for cursor_type")
        validate_boolean("allow_partial_results", allow_partial_results)
        validate_boolean("oplog_replay", oplog_replay)
        if modifiers is not None:
            warnings.warn("the 'modifiers' parameter is deprecated",
                          DeprecationWarning, stacklevel=2)
            validate_is_mapping("modifiers", modifiers)
        if not isinstance(batch_size, integer_types):
            raise TypeError("batch_size must be an integer")
        if batch_size < 0:
            raise ValueError("batch_size must be >= 0")
        if projection is not None:
            if not projection:
                projection = {"_id": 1}
            projection = helpers._fields_list_to_dict(projection, "projection")
        self.__collection = collection
        self.__spec = spec
        self.__projection = projection
        self.__skip = skip
        self.__limit = limit
        self.__batch_size = batch_size
        self.__modifiers = modifiers and modifiers.copy() or {}
        self.__ordering = sort and helpers._index_document(sort) or None
        self.__max_scan = max_scan
        self.__explain = False
        self.__comment = comment
        self.__max_time_ms = max_time_ms
        self.__max_await_time_ms = None
        self.__max = max
        self.__min = min
        self.__manipulate = manipulate
        self.__collation = validate_collation_or_none(collation)
        self.__return_key = return_key
        self.__show_record_id = show_record_id
        self.__snapshot = snapshot
        self.__set_hint(hint)
        if cursor_type == CursorType.EXHAUST:
            if self.__collection.database.client.is_mongos:
                raise InvalidOperation('Exhaust cursors are '
                                       'not supported by mongos')
            if limit:
                raise InvalidOperation("Can't use limit and exhaust together.")
            self.__exhaust = True
        self.__empty = False
        self.__data = deque()
        self.__address = None
        self.__retrieved = 0
        self.__killed = False
        self.__codec_options = collection.codec_options
        self.__read_preference = collection.read_preference
        self.__read_concern = collection.read_concern
        self.__query_flags = cursor_type
        if self.__read_preference != ReadPreference.PRIMARY:
            self.__query_flags |= _QUERY_OPTIONS["slave_okay"]
        if no_cursor_timeout:
            self.__query_flags |= _QUERY_OPTIONS["no_timeout"]
        if allow_partial_results:
            self.__query_flags |= _QUERY_OPTIONS["partial"]
        if oplog_replay:
            self.__query_flags |= _QUERY_OPTIONS["oplog_replay"]
    @property
    def collection(self):
        return self.__collection
    @property
    def retrieved(self):
        return self.__retrieved
    def __del__(self):
        self.__die()
    def rewind(self):
        self.__data = deque()
        self.__id = None
        self.__address = None
        self.__retrieved = 0
        self.__killed = False
        return self
    def clone(self):
        return self._clone(True)
    def _clone(self, deepcopy=True):
        clone = self._clone_base()
        values_to_clone = ("spec", "projection", "skip", "limit",
                           "max_time_ms", "max_await_time_ms", "comment",
                           "max", "min", "ordering", "explain", "hint",
                           "batch_size", "max_scan", "manipulate",
                           "query_flags", "modifiers", "collation")
        data = dict((k, v) for k, v in iteritems(self.__dict__)
                    if k.startswith('_Cursor__') and k[9:] in values_to_clone)
        if deepcopy:
            data = self._deepcopy(data)
        clone.__dict__.update(data)
        return clone
    def _clone_base(self):
        return Cursor(self.__collection)
    def __die(self, synchronous=False):
        if self.__id and not self.__killed:
            if self.__exhaust and self.__exhaust_mgr:
                self.__exhaust_mgr.sock.close()
            else:
                address = _CursorAddress(
                    self.__address, self.__collection.full_name)
                if synchronous:
                    self.__collection.database.client._close_cursor_now(
                        self.__id, address)
                else:
                    self.__collection.database.client.close_cursor(
                        self.__id, address)
        if self.__exhaust and self.__exhaust_mgr:
            self.__exhaust_mgr.close()
        self.__killed = True
    def close(self):
        self.__die(True)
    def __query_spec(self):
        operators = self.__modifiers.copy()
        if self.__ordering:
            operators["$orderby"] = self.__ordering
        if self.__explain:
            operators["$explain"] = True
        if self.__hint:
            operators["$hint"] = self.__hint
        if self.__comment:
            operators["$comment"] = self.__comment
        if self.__max_scan:
            operators["$maxScan"] = self.__max_scan
        if self.__max_time_ms is not None:
            operators["$maxTimeMS"] = self.__max_time_ms
        if self.__max:
            operators["$max"] = self.__max
        if self.__min:
            operators["$min"] = self.__min
        if self.__return_key:
            operators["$returnKey"] = self.__return_key
        if self.__show_record_id:
            operators["$showDiskLoc"] = self.__show_record_id
        if self.__snapshot:
            operators["$snapshot"] = self.__snapshot
        if operators:
            spec = self.__spec.copy()
            if "$query" not in spec:
                spec = SON([("$query", spec)])
            if not isinstance(spec, SON):
                spec = SON(spec)
            spec.update(operators)
            return spec
        elif ("query" in self.__spec and
              (len(self.__spec) == 1 or
               next(iter(self.__spec)) == "query")):
            return SON({"$query": self.__spec})
        return self.__spec
    def __check_okay_to_chain(self):
        if self.__retrieved or self.__id is not None:
            raise InvalidOperation("cannot set options after executing query")
    def add_option(self, mask):
        if not isinstance(mask, int):
            raise TypeError("mask must be an int")
        self.__check_okay_to_chain()
        if mask & _QUERY_OPTIONS["exhaust"]:
            if self.__limit:
                raise InvalidOperation("Can't use limit and exhaust together.")
            if self.__collection.database.client.is_mongos:
                raise InvalidOperation('Exhaust cursors are '
                                       'not supported by mongos')
            self.__exhaust = True
        self.__query_flags |= mask
        return self
    def remove_option(self, mask):
        if not isinstance(mask, int):
            raise TypeError("mask must be an int")
        self.__check_okay_to_chain()
        if mask & _QUERY_OPTIONS["exhaust"]:
            self.__exhaust = False
        self.__query_flags &= ~mask
        return self
    def limit(self, limit):
        if not isinstance(limit, integer_types):
            raise TypeError("limit must be an integer")
        if self.__exhaust:
            raise InvalidOperation("Can't use limit and exhaust together.")
        self.__check_okay_to_chain()
        self.__empty = False
        self.__limit = limit
        return self
    def batch_size(self, batch_size):
        if not isinstance(batch_size, integer_types):
            raise TypeError("batch_size must be an integer")
        if batch_size < 0:
            raise ValueError("batch_size must be >= 0")
        self.__check_okay_to_chain()
        self.__batch_size = batch_size
        return self
    def skip(self, skip):
        if not isinstance(skip, integer_types):
            raise TypeError("skip must be an integer")
        if skip < 0:
            raise ValueError("skip must be >= 0")
        self.__check_okay_to_chain()
        self.__skip = skip
        return self
    def max_time_ms(self, max_time_ms):
        if (not isinstance(max_time_ms, integer_types)
                and max_time_ms is not None):
            raise TypeError("max_time_ms must be an integer or None")
        self.__check_okay_to_chain()
        self.__max_time_ms = max_time_ms
        return self
    def max_await_time_ms(self, max_await_time_ms):
        if (not isinstance(max_await_time_ms, integer_types)
                and max_await_time_ms is not None):
            raise TypeError("max_await_time_ms must be an integer or None")
        self.__check_okay_to_chain()
        if self.__query_flags & CursorType.TAILABLE_AWAIT:
            self.__max_await_time_ms = max_await_time_ms
        return self
    def __getitem__(self, index):
        self.__check_okay_to_chain()
        self.__empty = False
        if isinstance(index, slice):
            if index.step is not None:
                raise IndexError("Cursor instances do not support slice steps")
            skip = 0
            if index.start is not None:
                if index.start < 0:
                    raise IndexError("Cursor instances do not support "
                                     "negative indices")
                skip = index.start
            if index.stop is not None:
                limit = index.stop - skip
                if limit < 0:
                    raise IndexError("stop index must be greater than start "
                                     "index for slice %r" % index)
                if limit == 0:
                    self.__empty = True
            else:
                limit = 0
            self.__skip = skip
            self.__limit = limit
            return self
        if isinstance(index, integer_types):
            if index < 0:
                raise IndexError("Cursor instances do not support negative "
                                 "indices")
            clone = self.clone()
            clone.skip(index + self.__skip)
            clone.limit(-1)
            for doc in clone:
                return doc
            raise IndexError("no such item for Cursor instance")
        raise TypeError("index %r cannot be applied to Cursor "
                        "instances" % index)
    def max_scan(self, max_scan):
        self.__check_okay_to_chain()
        self.__max_scan = max_scan
        return self
    def max(self, spec):
        if not isinstance(spec, (list, tuple)):
            raise TypeError("spec must be an instance of list or tuple")
        self.__check_okay_to_chain()
        self.__max = SON(spec)
        return self
    def min(self, spec):
        if not isinstance(spec, (list, tuple)):
            raise TypeError("spec must be an instance of list or tuple")
        self.__check_okay_to_chain()
        self.__min = SON(spec)
        return self
    def sort(self, key_or_list, direction=None):
        self.__check_okay_to_chain()
        keys = helpers._index_list(key_or_list, direction)
        self.__ordering = helpers._index_document(keys)
        return self
    def count(self, with_limit_and_skip=False):
        validate_boolean("with_limit_and_skip", with_limit_and_skip)
        cmd = SON([("count", self.__collection.name),
                   ("query", self.__spec)])
        if self.__max_time_ms is not None:
            cmd["maxTimeMS"] = self.__max_time_ms
        if self.__comment:
            cmd["$comment"] = self.__comment
        if self.__hint is not None:
            cmd["hint"] = self.__hint
        if with_limit_and_skip:
            if self.__limit:
                cmd["limit"] = self.__limit
            if self.__skip:
                cmd["skip"] = self.__skip
        return self.__collection._count(cmd, self.__collation)
    def distinct(self, key):
        options = {}
        if self.__spec:
            options["query"] = self.__spec
        if self.__max_time_ms is not None:
            options['maxTimeMS'] = self.__max_time_ms
        if self.__comment:
            options['$comment'] = self.__comment
        if self.__collation is not None:
            options['collation'] = self.__collation
        return self.__collection.distinct(key, **options)
    def explain(self):
        c = self.clone()
        c.__explain = True
        if c.__limit:
            c.__limit = -abs(c.__limit)
        return next(c)
    def __set_hint(self, index):
        if index is None:
            self.__hint = None
            return
        if isinstance(index, string_type):
            self.__hint = index
        else:
            self.__hint = helpers._index_document(index)
    def hint(self, index):
        self.__check_okay_to_chain()
        self.__set_hint(index)
        return self
    def comment(self, comment):
        self.__check_okay_to_chain()
        self.__comment = comment
        return self
    def where(self, code):
        self.__check_okay_to_chain()
        if not isinstance(code, Code):
            code = Code(code)
        self.__spec["$where"] = code
        return self
    def collation(self, collation):
        self.__check_okay_to_chain()
        self.__collation = validate_collation_or_none(collation)
        return self
    def __send_message(self, operation):
        client = self.__collection.database.client
        listeners = client._event_listeners
        publish = listeners.enabled_for_commands
        from_command = False
        if operation:
            kwargs = {
                "read_preference": self.__read_preference,
                "exhaust": self.__exhaust,
            }
            if self.__address is not None:
                kwargs["address"] = self.__address
            try:
                response = client._send_message_with_response(operation,
                                                              **kwargs)
                self.__address = response.address
                if self.__exhaust:
                    self.__exhaust_mgr = _SocketManager(response.socket_info,
                                                        response.pool)
                cmd_name = operation.name
                data = response.data
                cmd_duration = response.duration
                rqst_id = response.request_id
                from_command = response.from_command
            except AutoReconnect:
                self.__killed = True
                raise
        else:
            rqst_id = 0
            cmd_name = 'getMore'
            if publish:
                cmd = SON([('getMore', self.__id),
                           ('collection', self.__collection.name)])
                if self.__batch_size:
                    cmd['batchSize'] = self.__batch_size
                if self.__max_time_ms:
                    cmd['maxTimeMS'] = self.__max_time_ms
                listeners.publish_command_start(
                    cmd, self.__collection.database.name, 0, self.__address)
                start = datetime.datetime.now()
            try:
                data = self.__exhaust_mgr.sock.receive_message(1, None)
            except Exception as exc:
                if publish:
                    duration = datetime.datetime.now() - start
                    listeners.publish_command_failure(
                        duration, _convert_exception(exc), cmd_name, rqst_id,
                        self.__address)
                if isinstance(exc, ConnectionFailure):
                    self.__die()
                raise
            if publish:
                cmd_duration = datetime.datetime.now() - start
        if publish:
            start = datetime.datetime.now()
        try:
            doc = helpers._unpack_response(response=data,
                                           cursor_id=self.__id,
                                           codec_options=self.__codec_options)
            if from_command:
                helpers._check_command_response(doc['data'][0])
        except OperationFailure as exc:
            self.__killed = True
            self.__die()
            if publish:
                duration = (datetime.datetime.now() - start) + cmd_duration
                listeners.publish_command_failure(
                    duration, exc.details, cmd_name, rqst_id, self.__address)
            if self.__query_flags & _QUERY_OPTIONS["tailable_cursor"]:
                return
            raise
        except NotMasterError as exc:
            self.__killed = True
            self.__die()
            if publish:
                duration = (datetime.datetime.now() - start) + cmd_duration
                listeners.publish_command_failure(
                    duration, exc.details, cmd_name, rqst_id, self.__address)
            client._reset_server_and_request_check(self.__address)
            raise
        except Exception as exc:
            if publish:
                duration = (datetime.datetime.now() - start) + cmd_duration
                listeners.publish_command_failure(
                    duration, _convert_exception(exc), cmd_name, rqst_id,
                    self.__address)
            raise
        if publish:
            duration = (datetime.datetime.now() - start) + cmd_duration
            if from_command:
                res = doc['data'][0]
            elif cmd_name == "explain":
                res = doc["data"][0] if doc["number_returned"] else {}
            else:
                res = {"cursor": {"id": doc["cursor_id"],
                                  "ns": self.__collection.full_name},
                       "ok": 1}
                if cmd_name == "find":
                    res["cursor"]["firstBatch"] = doc["data"]
                else:
                    res["cursor"]["nextBatch"] = doc["data"]
            listeners.publish_command_success(
                duration, res, cmd_name, rqst_id, self.__address)
        if from_command and cmd_name != "explain":
            cursor = doc['data'][0]['cursor']
            self.__id = cursor['id']
            if cmd_name == 'find':
                documents = cursor['firstBatch']
            else:
                documents = cursor['nextBatch']
            self.__data = deque(documents)
            self.__retrieved += len(documents)
        else:
            self.__id = doc["cursor_id"]
            self.__data = deque(doc["data"])
            self.__retrieved += doc["number_returned"]
        if self.__id == 0:
            self.__killed = True
        if self.__limit and self.__id and self.__limit <= self.__retrieved:
            self.__die()
        if self.__exhaust and self.__id == 0:
            self.__exhaust_mgr.close()
    def _refresh(self):
        if len(self.__data) or self.__killed:
            return len(self.__data)
        if self.__id is None:
            self.__send_message(_Query(self.__query_flags,
                                       self.__collection.database.name,
                                       self.__collection.name,
                                       self.__skip,
                                       self.__query_spec(),
                                       self.__projection,
                                       self.__codec_options,
                                       self.__read_preference,
                                       self.__limit,
                                       self.__batch_size,
                                       self.__read_concern,
                                       self.__collation))
            if not self.__id:
                self.__killed = True
        elif self.__id:
            if self.__limit:
                limit = self.__limit - self.__retrieved
                if self.__batch_size:
                    limit = min(limit, self.__batch_size)
            else:
                limit = self.__batch_size
            if self.__exhaust:
                self.__send_message(None)
            else:
                self.__send_message(_GetMore(self.__collection.database.name,
                                             self.__collection.name,
                                             limit,
                                             self.__id,
                                             self.__codec_options,
                                             self.__max_await_time_ms))
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
        if self.__empty:
            raise StopIteration
        if len(self.__data) or self._refresh():
            if self.__manipulate:
                _db = self.__collection.database
                return _db._fix_outgoing(self.__data.popleft(),
                                         self.__collection)
            else:
                return self.__data.popleft()
        else:
            raise StopIteration
    __next__ = next
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    def __copy__(self):
        return self._clone(deepcopy=False)
    def __deepcopy__(self, memo):
        return self._clone(deepcopy=True)
    def _deepcopy(self, x, memo=None):
        if not hasattr(x, 'items'):
            y, is_list, iterator = [], True, enumerate(x)
        else:
            y, is_list, iterator = {}, False, iteritems(x)
        if memo is None:
            memo = {}
        val_id = id(x)
        if val_id in memo:
            return memo.get(val_id)
        memo[val_id] = y
        for key, value in iterator:
            if isinstance(value, (dict, list)) and not isinstance(value, SON):
                value = self._deepcopy(value, memo)
            elif not isinstance(value, RE_TYPE):
                value = copy.deepcopy(value, memo)
            if is_list:
                y.append(value)
            else:
                if not isinstance(key, RE_TYPE):
                    key = copy.deepcopy(key, memo)
                y[key] = value
        return y
