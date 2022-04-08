
"""Tools for connecting to MongoDB.
.. seealso:: :doc:`/examples/high_availability` for examples of connecting
   to replica sets or sets of mongos servers.
To get a :class:`~pymongo.database.Database` instance from a
:class:`MongoClient` use either dictionary-style or attribute-style
access:
.. doctest::
  >>> from pymongo import MongoClient
  >>> c = MongoClient()
  >>> c.test_database
  Database(MongoClient(host=['localhost:27017'], document_class=dict, tz_aware=False, connect=True), u'test_database')
  >>> c['test-database']
  Database(MongoClient(host=['localhost:27017'], document_class=dict, tz_aware=False, connect=True), u'test-database')
"""
import contextlib
import datetime
import threading
import warnings
import weakref
from collections import defaultdict
from bson.codec_options import DEFAULT_CODEC_OPTIONS
from bson.py3compat import (integer_types,
                            string_type)
from bson.son import SON
from pymongo import (common,
                     database,
                     helpers,
                     message,
                     periodic_executor,
                     uri_parser)
from pymongo.client_options import ClientOptions
from pymongo.cursor_manager import CursorManager
from pymongo.errors import (AutoReconnect,
                            ConfigurationError,
                            ConnectionFailure,
                            InvalidOperation,
                            InvalidURI,
                            NetworkTimeout,
                            NotMasterError,
                            OperationFailure)
from pymongo.read_preferences import ReadPreference
from pymongo.server_selectors import (writable_preferred_server_selector,
                                      writable_server_selector)
from pymongo.server_type import SERVER_TYPE
from pymongo.topology import Topology
from pymongo.topology_description import TOPOLOGY_TYPE
from pymongo.settings import TopologySettings
from pymongo.write_concern import WriteConcern
class MongoClient(common.BaseObject):
    HOST = "localhost"
    PORT = 27017
    _constructor_args = ('document_class', 'tz_aware', 'connect')
    def __init__(
            self,
            host=None,
            port=None,
            document_class=dict,
            tz_aware=None,
            connect=None,
            **kwargs):
        if host is None:
            host = self.HOST
        if isinstance(host, string_type):
            host = [host]
        if port is None:
            port = self.PORT
        if not isinstance(port, int):
            raise TypeError("port must be an instance of int")
        seeds = set()
        username = None
        password = None
        dbase = None
        opts = {}
        for entity in host:
            if "://" in entity:
                if entity.startswith("mongodb://"):
                    res = uri_parser.parse_uri(entity, port, warn=True)
                    seeds.update(res["nodelist"])
                    username = res["username"] or username
                    password = res["password"] or password
                    dbase = res["database"] or dbase
                    opts = res["options"]
                else:
                    idx = entity.find("://")
                    raise InvalidURI("Invalid URI scheme: "
                                     "%s" % (entity[:idx],))
            else:
                seeds.update(uri_parser.split_hosts(entity, port))
        if not seeds:
            raise ConfigurationError("need to specify at least one host")
        pool_class = kwargs.pop('_pool_class', None)
        monitor_class = kwargs.pop('_monitor_class', None)
        condition_class = kwargs.pop('_condition_class', None)
        keyword_opts = kwargs
        keyword_opts['document_class'] = document_class
        if tz_aware is None:
            tz_aware = opts.get('tz_aware', False)
        if connect is None:
            connect = opts.get('connect', True)
        keyword_opts['tz_aware'] = tz_aware
        keyword_opts['connect'] = connect
        keyword_opts = dict(common.validate(k, v)
                            for k, v in keyword_opts.items())
        opts.update(keyword_opts)
        username = opts.get("username", username)
        password = opts.get("password", password)
        if 'socketkeepalive' in opts:
            warnings.warn(
                "The socketKeepAlive option is deprecated. It now"
                "defaults to true and disabling it is not recommended, see "
                "https://docs.mongodb.com/manual/faq/diagnostics/"
                DeprecationWarning, stacklevel=2)
        self.__options = options = ClientOptions(
            username, password, dbase, opts)
        self.__default_database_name = dbase
        self.__lock = threading.Lock()
        self.__cursor_manager = None
        self.__kill_cursors_queue = []
        self._event_listeners = options.pool_options.event_listeners
        self.__index_cache = {}
        self.__index_cache_lock = threading.Lock()
        super(MongoClient, self).__init__(options.codec_options,
                                          options.read_preference,
                                          options.write_concern,
                                          options.read_concern)
        self.__all_credentials = {}
        creds = options.credentials
        if creds:
            self._cache_credentials(creds.source, creds)
        self._topology_settings = TopologySettings(
            seeds=seeds,
            replica_set_name=options.replica_set_name,
            pool_class=pool_class,
            pool_options=options.pool_options,
            monitor_class=monitor_class,
            condition_class=condition_class,
            local_threshold_ms=options.local_threshold_ms,
            server_selection_timeout=options.server_selection_timeout,
            heartbeat_frequency=options.heartbeat_frequency)
        self._topology = Topology(self._topology_settings)
        if connect:
            self._topology.open()
        def target():
            client = self_ref()
            if client is None:
                return False
            MongoClient._process_periodic_tasks(client)
            return True
        executor = periodic_executor.PeriodicExecutor(
            interval=common.KILL_CURSOR_FREQUENCY,
            min_interval=0.5,
            target=target,
            name="pymongo_kill_cursors_thread")
        self_ref = weakref.ref(self, executor.close)
        self._kill_cursors_executor = executor
        executor.open()
    def _cache_credentials(self, source, credentials, connect=False):
        all_credentials = self.__all_credentials.copy()
        if source in all_credentials:
            if credentials == all_credentials[source]:
                return
            raise OperationFailure('Another user is already authenticated '
                                   'to this database. You must logout first.')
        if connect:
            server = self._get_topology().select_server(
                writable_preferred_server_selector)
            with server.get_socket(all_credentials) as sock_info:
                sock_info.authenticate(credentials)
        self.__all_credentials[source] = credentials
    def _purge_credentials(self, source):
        self.__all_credentials.pop(source, None)
    def _cached(self, dbname, coll, index):
        cache = self.__index_cache
        now = datetime.datetime.utcnow()
        with self.__index_cache_lock:
            return (dbname in cache and
                    coll in cache[dbname] and
                    index in cache[dbname][coll] and
                    now < cache[dbname][coll][index])
    def _cache_index(self, dbname, collection, index, cache_for):
        now = datetime.datetime.utcnow()
        expire = datetime.timedelta(seconds=cache_for) + now
        with self.__index_cache_lock:
            if database not in self.__index_cache:
                self.__index_cache[dbname] = {}
                self.__index_cache[dbname][collection] = {}
                self.__index_cache[dbname][collection][index] = expire
            elif collection not in self.__index_cache[dbname]:
                self.__index_cache[dbname][collection] = {}
                self.__index_cache[dbname][collection][index] = expire
            else:
                self.__index_cache[dbname][collection][index] = expire
    def _purge_index(self, database_name,
                     collection_name=None, index_name=None):
        with self.__index_cache_lock:
            if not database_name in self.__index_cache:
                return
            if collection_name is None:
                del self.__index_cache[database_name]
                return
            if not collection_name in self.__index_cache[database_name]:
                return
            if index_name is None:
                del self.__index_cache[database_name][collection_name]
                return
            if index_name in self.__index_cache[database_name][collection_name]:
                del self.__index_cache[database_name][collection_name][index_name]
    def _server_property(self, attr_name):
        server = self._topology.select_server(
            writable_server_selector)
        return getattr(server.description, attr_name)
    @property
    def event_listeners(self):
        return self._event_listeners.event_listeners
    @property
    def address(self):
        topology_type = self._topology._description.topology_type
        if topology_type == TOPOLOGY_TYPE.Sharded:
            raise InvalidOperation(
                'Cannot use "address" property when load balancing among'
                ' mongoses, use "nodes" instead.')
        if topology_type not in (TOPOLOGY_TYPE.ReplicaSetWithPrimary,
                                 TOPOLOGY_TYPE.Single):
            return None
        return self._server_property('address')
    @property
    def primary(self):
        return self._topology.get_primary()
    @property
    def secondaries(self):
        return self._topology.get_secondaries()
    @property
    def arbiters(self):
        return self._topology.get_arbiters()
    @property
    def is_primary(self):
        return self._server_property('is_writable')
    @property
    def is_mongos(self):
        return self._server_property('server_type') == SERVER_TYPE.Mongos
    @property
    def max_pool_size(self):
        return self.__options.pool_options.max_pool_size
    @property
    def min_pool_size(self):
        return self.__options.pool_options.min_pool_size
    @property
    def max_idle_time_ms(self):
        return self.__options.pool_options.max_idle_time_ms
    @property
    def nodes(self):
        description = self._topology.description
        return frozenset(s.address for s in description.known_servers)
    @property
    def max_bson_size(self):
        return self._server_property('max_bson_size')
    @property
    def max_message_size(self):
        return self._server_property('max_message_size')
    @property
    def max_write_batch_size(self):
        return self._server_property('max_write_batch_size')
    @property
    def local_threshold_ms(self):
        return self.__options.local_threshold_ms
    @property
    def server_selection_timeout(self):
        return self.__options.server_selection_timeout
    def _is_writable(self):
        topology = self._get_topology()
        try:
            svr = topology.select_server(writable_server_selector)
            return svr.description.is_writable
        except ConnectionFailure:
            return False
    def close(self):
        self._process_periodic_tasks()
        self._topology.close()
    def set_cursor_manager(self, manager_class):
        warnings.warn(
            "set_cursor_manager is Deprecated",
            DeprecationWarning,
            stacklevel=2)
        manager = manager_class(self)
        if not isinstance(manager, CursorManager):
            raise TypeError("manager_class must be a subclass of "
                            "CursorManager")
        self.__cursor_manager = manager
    def _get_topology(self):
        self._topology.open()
        return self._topology
    @contextlib.contextmanager
    def _get_socket(self, selector):
        server = self._get_topology().select_server(selector)
        try:
            with server.get_socket(self.__all_credentials) as sock_info:
                yield sock_info
        except NetworkTimeout:
            raise
        except NotMasterError:
            self._reset_server_and_request_check(server.description.address)
            raise
        except ConnectionFailure:
            self.__reset_server(server.description.address)
            raise
    def _socket_for_writes(self):
        return self._get_socket(writable_server_selector)
    @contextlib.contextmanager
    def _socket_for_reads(self, read_preference):
        preference = read_preference or ReadPreference.PRIMARY
        topology = self._get_topology()
        single = topology.description.topology_type == TOPOLOGY_TYPE.Single
        with self._get_socket(read_preference) as sock_info:
            slave_ok = (single and not sock_info.is_mongos) or (
                preference != ReadPreference.PRIMARY)
            yield sock_info, slave_ok
    def _send_message_with_response(self, operation, read_preference=None,
                                    exhaust=False, address=None):
        with self.__lock:
            self._kill_cursors_executor.open()
        topology = self._get_topology()
        if address:
            server = topology.select_server_by_address(address)
            if not server:
                raise AutoReconnect('server %s:%d no longer available'
                                    % address)
        else:
            selector = read_preference or writable_server_selector
            server = topology.select_server(selector)
        set_slave_ok = (
            topology.description.topology_type == TOPOLOGY_TYPE.Single
            and server.description.server_type != SERVER_TYPE.Mongos)
        return self._reset_on_error(
            server,
            server.send_message_with_response,
            operation,
            set_slave_ok,
            self.__all_credentials,
            self._event_listeners,
            exhaust)
    def _reset_on_error(self, server, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NetworkTimeout:
            raise
        except ConnectionFailure:
            self.__reset_server(server.description.address)
            raise
    def __reset_server(self, address):
        self._topology.reset_server(address)
    def _reset_server_and_request_check(self, address):
        self._topology.reset_server_and_request_check(address)
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.address == other.address
        return NotImplemented
    def __ne__(self, other):
        return not self == other
    def _repr_helper(self):
        def option_repr(option, value):
            if option == 'document_class':
                if value is dict:
                    return 'document_class=dict'
                else:
                    return 'document_class=%s.%s' % (value.__module__,
                                                     value.__name__)
            if option in common.TIMEOUT_VALIDATORS and value is not None:
                return "%s=%s" % (option, int(value * 1000))
            return '%s=%r' % (option, value)
        options = ['host=%r' % [
            '%s:%d' % (host, port) if port is not None else host
            for host, port in self._topology_settings.seeds]]
        options.extend(
            option_repr(key, self.__options._options[key])
            for key in self._constructor_args)
        options.extend(
            option_repr(key, self.__options._options[key])
            for key in self.__options._options
            if key not in set(self._constructor_args)
            and key != 'username' and key != 'password')
        return ', '.join(options)
    def __repr__(self):
        return ("MongoClient(%s)" % (self._repr_helper(),))
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(
                "MongoClient has no attribute %r. To access the %s"
                " database, use client[%r]." % (name, name, name))
        return self.__getitem__(name)
    def __getitem__(self, name):
        return database.Database(self, name)
    def close_cursor(self, cursor_id, address=None):
        if not isinstance(cursor_id, integer_types):
            raise TypeError("cursor_id must be an instance of (int, long)")
        if self.__cursor_manager is not None:
            self.__cursor_manager.close(cursor_id, address)
        else:
            self.__kill_cursors_queue.append((address, [cursor_id]))
    def _close_cursor_now(self, cursor_id, address=None):
        if not isinstance(cursor_id, integer_types):
            raise TypeError("cursor_id must be an instance of (int, long)")
        if self.__cursor_manager is not None:
            self.__cursor_manager.close(cursor_id, address)
        else:
            self._kill_cursors([cursor_id], address, self._get_topology())
    def kill_cursors(self, cursor_ids, address=None):
        warnings.warn(
            "kill_cursors is deprecated.",
            DeprecationWarning,
            stacklevel=2)
        if not isinstance(cursor_ids, list):
            raise TypeError("cursor_ids must be a list")
        self.__kill_cursors_queue.append((address, cursor_ids))
    def _kill_cursors(self, cursor_ids, address, topology):
        listeners = self._event_listeners
        publish = listeners.enabled_for_commands
        if address:
            server = topology.select_server_by_address(tuple(address))
        else:
            server = topology.select_server(writable_server_selector)
        try:
            namespace = address.namespace
            db, coll = namespace.split('.', 1)
        except AttributeError:
            namespace = None
            db = coll = "OP_KILL_CURSORS"
        spec = SON([('killCursors', coll), ('cursors', cursor_ids)])
        with server.get_socket(self.__all_credentials) as sock_info:
            if sock_info.max_wire_version >= 4 and namespace is not None:
                sock_info.command(db, spec)
            else:
                if publish:
                    start = datetime.datetime.now()
                request_id, msg = message.kill_cursors(cursor_ids)
                if publish:
                    duration = datetime.datetime.now() - start
                    listeners.publish_command_start(
                        spec, db, request_id, tuple(address))
                    start = datetime.datetime.now()
                try:
                    sock_info.send_message(msg, 0)
                except Exception as exc:
                    if publish:
                        dur = ((datetime.datetime.now() - start) + duration)
                        listeners.publish_command_failure(
                            dur, message._convert_exception(exc),
                            'killCursors', request_id,
                            tuple(address))
                    raise
                if publish:
                    duration = ((datetime.datetime.now() - start) + duration)
                    reply = {'cursorsUnknown': cursor_ids, 'ok': 1}
                    listeners.publish_command_success(
                        duration, reply, 'killCursors', request_id,
                        tuple(address))
    def _process_periodic_tasks(self):
        address_to_cursor_ids = defaultdict(list)
        while True:
            try:
                address, cursor_ids = self.__kill_cursors_queue.pop()
            except IndexError:
                break
            address_to_cursor_ids[address].extend(cursor_ids)
        if address_to_cursor_ids:
            topology = self._get_topology()
            for address, cursor_ids in address_to_cursor_ids.items():
                try:
                    self._kill_cursors(cursor_ids, address, topology)
                except Exception:
                    helpers._handle_exception()
        try:
            self._topology.update_pool()
        except Exception:
            helpers._handle_exception()
    def server_info(self):
        return self.admin.command("buildinfo",
                                  read_preference=ReadPreference.PRIMARY)
    def database_names(self):
        return [db["name"] for db in
                self._database_default_options("admin").command(
                    SON([("listDatabases", 1),
                         ("nameOnly", True)]))["databases"]]
    def drop_database(self, name_or_database):
        name = name_or_database
        if isinstance(name, database.Database):
            name = name.name
        if not isinstance(name, string_type):
            raise TypeError("name_or_database must be an instance "
                            "of %s or a Database" % (string_type.__name__,))
        self._purge_index(name)
        with self._socket_for_reads(
                ReadPreference.PRIMARY) as (sock_info, slave_ok):
            self[name]._command(
                sock_info,
                "dropDatabase",
                slave_ok=slave_ok,
                read_preference=ReadPreference.PRIMARY,
                write_concern=self.write_concern,
                parse_write_concern_error=True)
    def get_default_database(self):
        warnings.warn("get_default_database is deprecated. Use get_database "
                      "instead.", DeprecationWarning, stacklevel=2)
        if self.__default_database_name is None:
            raise ConfigurationError('No default database defined')
        return self[self.__default_database_name]
    def get_database(self, name=None, codec_options=None, read_preference=None,
                     write_concern=None, read_concern=None):
        if name is None:
            if self.__default_database_name is None:
                raise ConfigurationError('No default database defined')
            name = self.__default_database_name
        return database.Database(
            self, name, codec_options, read_preference,
            write_concern, read_concern)
    def _database_default_options(self, name):
        return self.get_database(
            name, codec_options=DEFAULT_CODEC_OPTIONS,
            read_preference=ReadPreference.PRIMARY,
            write_concern=WriteConcern())
    @property
    def is_locked(self):
        ops = self._database_default_options('admin').current_op()
        return bool(ops.get('fsyncLock', 0))
    def fsync(self, **kwargs):
        self.admin.command("fsync",
                           read_preference=ReadPreference.PRIMARY, **kwargs)
    def unlock(self):
        cmd = {"fsyncUnlock": 1}
        with self._socket_for_writes() as sock_info:
            if sock_info.max_wire_version >= 4:
                try:
                    sock_info.command("admin", cmd)
                except OperationFailure as exc:
                    if exc.code != 125:
                        raise
            else:
                helpers._first_batch(sock_info, "admin", "$cmd.sys.unlock",
                    {}, -1, True, self.codec_options,
                    ReadPreference.PRIMARY, cmd, self._event_listeners)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    def __iter__(self):
        return self
    def __next__(self):
        raise TypeError("'MongoClient' object is not iterable")
    next = __next__
