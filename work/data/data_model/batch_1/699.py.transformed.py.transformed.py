
import weakref
from bson.codec_options import DEFAULT_CODEC_OPTIONS
from bson.son import SON
from pymongo import common, helpers, message, periodic_executor
from pymongo.server_type import SERVER_TYPE
from pymongo.ismaster import IsMaster
from pymongo.monotonic import time as _time
from pymongo.read_preferences import MovingAverage
from pymongo.server_description import ServerDescription
class Monitor(object):
    def __init__(
            self,
            server_description,
            topology,
            pool,
            topology_settings):
        self._server_description = server_description
        self._pool = pool
        self._settings = topology_settings
        self._avg_round_trip_time = MovingAverage()
        self._listeners = self._settings._pool_options.event_listeners
        pub = self._listeners is not None
        self._publish = pub and self._listeners.enabled_for_server_heartbeat
        def target():
            monitor = self_ref()
            if monitor is None:
                return False
            Monitor._run(monitor)
            return True
        executor = periodic_executor.PeriodicExecutor(
            interval=self._settings.heartbeat_frequency,
            min_interval=common.MIN_HEARTBEAT_INTERVAL,
            target=target,
            name="pymongo_server_monitor_thread")
        self._executor = executor
        self_ref = weakref.ref(self, executor.close)
        self._topology = weakref.proxy(topology, executor.close)
    def open(self):
        self._executor.open()
    def close(self):
        self._executor.close()
        self._pool.reset()
    def join(self, timeout=None):
        self._executor.join(timeout)
    def request_check(self):
        self._executor.wake()
    def _run(self):
        try:
            self._server_description = self._check_with_retry()
            self._topology.on_change(self._server_description)
        except ReferenceError:
            self.close()
    def _check_with_retry(self):
        address = self._server_description.address
        retry = True
        metadata = None
        if self._server_description.server_type == SERVER_TYPE.Unknown:
            retry = False
            metadata = self._pool.opts.metadata
        start = _time()
        try:
            return self._check_once(metadata=metadata)
        except ReferenceError:
            raise
        except Exception as error:
            error_time = _time() - start
            if self._publish:
                self._listeners.publish_server_heartbeat_failed(
                    address, error_time, error)
            self._topology.reset_pool(address)
            default = ServerDescription(address, error=error)
            if not retry:
                self._avg_round_trip_time.reset()
                return default
            start = _time()
            try:
                return self._check_once(metadata=self._pool.opts.metadata)
            except ReferenceError:
                raise
            except Exception as error:
                error_time = _time() - start
                if self._publish:
                    self._listeners.publish_server_heartbeat_failed(
                        address, error_time, error)
                self._avg_round_trip_time.reset()
                return default
    def _check_once(self, metadata=None):
        address = self._server_description.address
        if self._publish:
            self._listeners.publish_server_heartbeat_started(address)
        with self._pool.get_socket({}) as sock_info:
            response, round_trip_time = self._check_with_socket(
                sock_info, metadata=metadata)
            self._avg_round_trip_time.add_sample(round_trip_time)
            sd = ServerDescription(
                address=address,
                ismaster=response,
                round_trip_time=self._avg_round_trip_time.get())
            if self._publish:
                self._listeners.publish_server_heartbeat_succeeded(
                    address, round_trip_time, response)
            return sd
    def _check_with_socket(self, sock_info, metadata=None):
        cmd = SON([('ismaster', 1)])
        if metadata is not None:
            cmd['client'] = metadata
        start = _time()
        request_id, msg, max_doc_size = message.query(
            0, 'admin.$cmd', 0, -1, cmd,
            None, DEFAULT_CODEC_OPTIONS)
        sock_info.send_message(msg, max_doc_size)
        raw_response = sock_info.receive_message(1, request_id)
        result = helpers._unpack_response(raw_response)
        return IsMaster(result['data'][0]), _time() - start
