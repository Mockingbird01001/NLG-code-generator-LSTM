
import datetime
import errno
import select
import struct
import threading
_HAS_POLL = True
_EVENT_MASK = 0
try:
    from select import poll
    _EVENT_MASK = (
        select.POLLIN | select.POLLPRI | select.POLLERR | select.POLLHUP)
except ImportError:
    _HAS_POLL = False
try:
    from select import error as _SELECT_ERROR
except ImportError:
    _SELECT_ERROR = OSError
from pymongo import helpers, message
from pymongo.common import MAX_MESSAGE_SIZE
from pymongo.errors import (AutoReconnect,
                            NotMasterError,
                            OperationFailure,
                            ProtocolError)
from pymongo.read_concern import DEFAULT_READ_CONCERN
_UNPACK_INT = struct.Struct("<i").unpack
def command(sock, dbname, spec, slave_ok, is_mongos,
            read_preference, codec_options, check=True,
            allowable_errors=None, address=None,
            check_keys=False, listeners=None, max_bson_size=None,
            read_concern=DEFAULT_READ_CONCERN,
            parse_write_concern_error=False,
            collation=None):
    name = next(iter(spec))
    ns = dbname + '.$cmd'
    flags = 4 if slave_ok else 0
    orig = spec
    if is_mongos:
        spec = message._maybe_add_read_preference(spec, read_preference)
    if read_concern.level:
        spec['readConcern'] = read_concern.document
    if collation is not None:
        spec['collation'] = collation
    publish = listeners is not None and listeners.enabled_for_commands
    if publish:
        start = datetime.datetime.now()
    request_id, msg, size = message.query(flags, ns, 0, -1, spec,
                                          None, codec_options, check_keys)
    if (max_bson_size is not None
            and size > max_bson_size + message._COMMAND_OVERHEAD):
        message._raise_document_too_large(
            name, size, max_bson_size + message._COMMAND_OVERHEAD)
    if publish:
        encoding_duration = datetime.datetime.now() - start
        listeners.publish_command_start(orig, dbname, request_id, address)
        start = datetime.datetime.now()
    try:
        sock.sendall(msg)
        response = receive_message(sock, 1, request_id)
        unpacked = helpers._unpack_response(
            response, codec_options=codec_options)
        response_doc = unpacked['data'][0]
        if check:
            helpers._check_command_response(
                response_doc, None, allowable_errors,
                parse_write_concern_error=parse_write_concern_error)
    except Exception as exc:
        if publish:
            duration = (datetime.datetime.now() - start) + encoding_duration
            if isinstance(exc, (NotMasterError, OperationFailure)):
                failure = exc.details
            else:
                failure = message._convert_exception(exc)
            listeners.publish_command_failure(
                duration, failure, name, request_id, address)
        raise
    if publish:
        duration = (datetime.datetime.now() - start) + encoding_duration
        listeners.publish_command_success(
            duration, response_doc, name, request_id, address)
    return response_doc
def receive_message(
        sock, operation, request_id, max_message_size=MAX_MESSAGE_SIZE):
    header = _receive_data_on_socket(sock, 16)
    length = _UNPACK_INT(header[:4])[0]
    actual_op = _UNPACK_INT(header[12:])[0]
    if operation != actual_op:
        raise ProtocolError("Got opcode %r but expected "
                            "%r" % (actual_op, operation))
    if request_id is not None:
        response_id = _UNPACK_INT(header[8:12])[0]
        if request_id != response_id:
            raise ProtocolError("Got response id %r but expected "
                                "%r" % (response_id, request_id))
    if length <= 16:
        raise ProtocolError("Message length (%r) not longer than standard "
                            "message header size (16)" % (length,))
    if length > max_message_size:
        raise ProtocolError("Message length (%r) is larger than server max "
                            "message size (%r)" % (length, max_message_size))
    return _receive_data_on_socket(sock, length - 16)
def _receive_data_on_socket(sock, length):
    msg = b""
    while length:
        try:
            chunk = sock.recv(length)
        except (IOError, OSError) as exc:
            if _errno_from_exception(exc) == errno.EINTR:
                continue
            raise
        if chunk == b"":
            raise AutoReconnect("connection closed")
        length -= len(chunk)
        msg += chunk
    return msg
def _errno_from_exception(exc):
    if hasattr(exc, 'errno'):
        return exc.errno
    elif exc.args:
        return exc.args[0]
    else:
        return None
class SocketChecker(object):
    def __init__(self):
        if _HAS_POLL:
            self._lock = threading.Lock()
            self._poller = poll()
        else:
            self._lock = None
            self._poller = None
    def socket_closed(self, sock):
        while True:
            try:
                if self._poller:
                    with self._lock:
                        self._poller.register(sock, _EVENT_MASK)
                        try:
                            rd = self._poller.poll(0)
                        finally:
                            self._poller.unregister(sock)
                else:
                    rd, _, _ = select.select([sock], [], [], 0)
            except (RuntimeError, KeyError):
                raise
            except ValueError:
                return True
            except (_SELECT_ERROR, IOError) as exc:
                if _errno_from_exception(exc) in (errno.EINTR, errno.EAGAIN):
                    continue
                return True
            except:
                return True
            return len(rd) > 0
