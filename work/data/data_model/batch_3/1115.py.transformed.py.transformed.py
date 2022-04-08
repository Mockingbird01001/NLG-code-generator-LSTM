
import os
import random
import threading
import warnings
import weakref
from bson.py3compat import itervalues, PY3
if PY3:
    import queue as Queue
else:
    import Queue
from pymongo import common
from pymongo import periodic_executor
from pymongo.pool import PoolOptions
from pymongo.topology_description import (updated_topology_description,
                                          TOPOLOGY_TYPE,
                                          TopologyDescription)
from pymongo.errors import ServerSelectionTimeoutError
from pymongo.monotonic import time as _time
from pymongo.server import Server
from pymongo.server_selectors import (any_server_selector,
                                      arbiter_server_selector,
                                      secondary_server_selector,
                                      writable_server_selector,
                                      Selection)
def process_events_queue(queue_ref):
    q = queue_ref()
    if not q:
        return False
    while True:
        try:
            event = q.get_nowait()
        except Queue.Empty:
            break
        else:
            fn, args = event
            fn(*args)
    return True
class Topology(object):
    def __init__(self, topology_settings):
        self._topology_id = topology_settings._topology_id
        self._listeners = topology_settings._pool_options.event_listeners
        pub = self._listeners is not None
        self._publish_server = pub and self._listeners.enabled_for_server
        self._publish_tp = pub and self._listeners.enabled_for_topology
        self._events = None
        self._events_thread = None
        if self._publish_server or self._publish_tp:
            self._events = Queue.Queue(maxsize=100)
        if self._publish_tp:
            self._events.put((self._listeners.publish_topology_opened,
                             (self._topology_id,)))
        self._settings = topology_settings
        topology_description = TopologyDescription(
            topology_settings.get_topology_type(),
            topology_settings.get_server_descriptions(),
            topology_settings.replica_set_name,
            None,
            None,
            topology_settings)
        self._description = topology_description
        if self._publish_tp:
            initial_td = TopologyDescription(TOPOLOGY_TYPE.Unknown, {}, None,
                                             None, None, self._settings)
            self._events.put((
                self._listeners.publish_topology_description_changed,
                (initial_td, self._description, self._topology_id)))
        for seed in topology_settings.seeds:
            if self._publish_server:
                self._events.put((self._listeners.publish_server_opened,
                                 (seed, self._topology_id)))
        self._seed_addresses = list(topology_description.server_descriptions())
        self._opened = False
        self._lock = threading.Lock()
        self._condition = self._settings.condition_class(self._lock)
        self._servers = {}
        self._pid = None
        if self._publish_server or self._publish_tp:
            def target():
                return process_events_queue(weak)
            executor = periodic_executor.PeriodicExecutor(
                interval=common.EVENTS_QUEUE_FREQUENCY,
                min_interval=0.5,
                target=target,
                name="pymongo_events_thread")
            weak = weakref.ref(self._events)
            self.__events_executor = executor
            executor.open()
    def open(self):
        if self._pid is None:
            self._pid = os.getpid()
        else:
            if os.getpid() != self._pid:
                warnings.warn(
                    "MongoClient opened before fork. Create MongoClient "
                    "with connect=False, or create client after forking. "
                    "See PyMongo's documentation for details: http://api."
        with self._lock:
            self._ensure_opened()
    def select_servers(self,
                       selector,
                       server_selection_timeout=None,
                       address=None):
        if server_selection_timeout is None:
            server_timeout = self._settings.server_selection_timeout
        else:
            server_timeout = server_selection_timeout
        with self._lock:
            self._description.check_compatible()
            now = _time()
            end_time = now + server_timeout
            server_descriptions = self._description.apply_selector(
                selector, address)
            while not server_descriptions:
                if server_timeout == 0 or now > end_time:
                    raise ServerSelectionTimeoutError(
                        self._error_message(selector))
                self._ensure_opened()
                self._request_check_all()
                self._condition.wait(common.MIN_HEARTBEAT_INTERVAL)
                self._description.check_compatible()
                now = _time()
                server_descriptions = self._description.apply_selector(
                    selector, address)
            return [self.get_server_by_address(sd.address)
                    for sd in server_descriptions]
    def select_server(self,
                      selector,
                      server_selection_timeout=None,
                      address=None):
        return random.choice(self.select_servers(selector,
                                                 server_selection_timeout,
                                                 address))
    def select_server_by_address(self, address,
                                 server_selection_timeout=None):
        return self.select_server(any_server_selector,
                                  server_selection_timeout,
                                  address)
    def on_change(self, server_description):
        with self._lock:
            if self._description.has_server(server_description.address):
                td_old = self._description
                if self._publish_server:
                    old_server_description = td_old._server_descriptions[
                        server_description.address]
                    self._events.put((
                        self._listeners.publish_server_description_changed,
                        (old_server_description, server_description,
                         server_description.address, self._topology_id)))
                self._description = updated_topology_description(
                    self._description, server_description)
                self._update_servers()
                if self._publish_tp:
                    self._events.put((
                        self._listeners.publish_topology_description_changed,
                        (td_old, self._description, self._topology_id)))
                self._condition.notify_all()
    def get_server_by_address(self, address):
        return self._servers.get(address)
    def has_server(self, address):
        return address in self._servers
    def get_primary(self):
        with self._lock:
            topology_type = self._description.topology_type
            if topology_type != TOPOLOGY_TYPE.ReplicaSetWithPrimary:
                return None
            return writable_server_selector(self._new_selection())[0].address
    def _get_replica_set_members(self, selector):
        with self._lock:
            topology_type = self._description.topology_type
            if topology_type not in (TOPOLOGY_TYPE.ReplicaSetWithPrimary,
                                     TOPOLOGY_TYPE.ReplicaSetNoPrimary):
                return set()
            return set([sd.address for sd in selector(self._new_selection())])
    def get_secondaries(self):
        return self._get_replica_set_members(secondary_server_selector)
    def get_arbiters(self):
        return self._get_replica_set_members(arbiter_server_selector)
    def request_check_all(self, wait_time=5):
        with self._lock:
            self._request_check_all()
            self._condition.wait(wait_time)
    def reset_pool(self, address):
        with self._lock:
            server = self._servers.get(address)
            if server:
                server.pool.reset()
    def reset_server(self, address):
        with self._lock:
            self._reset_server(address)
    def reset_server_and_request_check(self, address):
        with self._lock:
            self._reset_server(address)
            self._request_check(address)
    def update_pool(self):
        with self._lock:
            for server in self._servers.values():
                server._pool.remove_stale_sockets()
    def close(self):
        with self._lock:
            for server in self._servers.values():
                server.close()
            self._description = self._description.reset()
            self._update_servers()
            self._opened = False
        if self._publish_tp:
            self._events.put((self._listeners.publish_topology_closed,
                              (self._topology_id,)))
        if self._publish_server or self._publish_tp:
            self.__events_executor.close()
    @property
    def description(self):
        return self._description
    def _new_selection(self):
        return Selection.from_topology_description(self._description)
    def _ensure_opened(self):
        if not self._opened:
            self._opened = True
            self._update_servers()
            if self._publish_tp or self._publish_server:
                self.__events_executor.open()
        else:
            for server in itervalues(self._servers):
                server.open()
    def _reset_server(self, address):
        server = self._servers.get(address)
        if server:
            server.reset()
            self._description = self._description.reset_server(address)
            self._update_servers()
    def _request_check(self, address):
        server = self._servers.get(address)
        if server:
            server.request_check()
    def _request_check_all(self):
        for server in self._servers.values():
            server.request_check()
    def _update_servers(self):
        for address, sd in self._description.server_descriptions().items():
            if address not in self._servers:
                monitor = self._settings.monitor_class(
                    server_description=sd,
                    topology=self,
                    pool=self._create_pool_for_monitor(address),
                    topology_settings=self._settings)
                weak = None
                if self._publish_server:
                    weak = weakref.ref(self._events)
                server = Server(
                    server_description=sd,
                    pool=self._create_pool_for_server(address),
                    monitor=monitor,
                    topology_id=self._topology_id,
                    listeners=self._listeners,
                    events=weak)
                self._servers[address] = server
                server.open()
            else:
                self._servers[address].description = sd
        for address, server in list(self._servers.items()):
            if not self._description.has_server(address):
                server.close()
                self._servers.pop(address)
    def _create_pool_for_server(self, address):
        return self._settings.pool_class(address, self._settings.pool_options)
    def _create_pool_for_monitor(self, address):
        options = self._settings.pool_options
        monitor_pool_options = PoolOptions(
            connect_timeout=options.connect_timeout,
            socket_timeout=options.connect_timeout,
            ssl_context=options.ssl_context,
            ssl_match_hostname=options.ssl_match_hostname,
            event_listeners=options.event_listeners,
            appname=options.appname)
        return self._settings.pool_class(address, monitor_pool_options,
                                         handshake=False)
    def _error_message(self, selector):
        is_replica_set = self._description.topology_type in (
            TOPOLOGY_TYPE.ReplicaSetWithPrimary,
            TOPOLOGY_TYPE.ReplicaSetNoPrimary)
        if is_replica_set:
            server_plural = 'replica set members'
        elif self._description.topology_type == TOPOLOGY_TYPE.Sharded:
            server_plural = 'mongoses'
        else:
            server_plural = 'servers'
        if self._description.known_servers:
            if selector is writable_server_selector:
                if is_replica_set:
                    return 'No primary available for writes'
                else:
                    return 'No %s available for writes' % server_plural
            else:
                return 'No %s match selector "%s"' % (server_plural, selector)
        else:
            addresses = list(self._description.server_descriptions())
            servers = list(self._description.server_descriptions().values())
            if not servers:
                if is_replica_set:
                    return 'No %s available for replica set name "%s"' % (
                        server_plural, self._settings.replica_set_name)
                else:
                    return 'No %s available' % server_plural
            error = servers[0].error
            same = all(server.error == error for server in servers[1:])
            if same:
                if error is None:
                    return 'No %s found yet' % server_plural
                if (is_replica_set and not
                        set(addresses).intersection(self._seed_addresses)):
                    return (
                        'Could not reach any servers in %s. Replica set is'
                        ' configured with internal hostnames or IPs?' %
                        addresses)
                return str(error)
            else:
                return ','.join(str(server.error) for server in servers
                                if server.error)
