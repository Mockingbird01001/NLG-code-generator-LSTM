
"""Tools to monitor driver events.
.. versionadded:: 3.1
Use :func:`register` to register global listeners for specific events.
Listeners must inherit from one of the abstract classes below and implement
the correct functions for that class.
For example, a simple command logger might be implemented like this::
    import logging
    from pymongo import monitoring
    class CommandLogger(monitoring.CommandListener):
        def started(self, event):
            logging.info("Command {0.command_name} with request id "
                         "{0.request_id} started on server "
                         "{0.connection_id}".format(event))
        def succeeded(self, event):
            logging.info("Command {0.command_name} with request id "
                         "{0.request_id} on server {0.connection_id} "
                         "succeeded in {0.duration_micros} "
                         "microseconds".format(event))
        def failed(self, event):
            logging.info("Command {0.command_name} with request id "
                         "{0.request_id} on server {0.connection_id} "
                         "failed in {0.duration_micros} "
                         "microseconds".format(event))
    monitoring.register(CommandLogger())
Server discovery and monitoring events are also available. For example::
    class ServerLogger(monitoring.ServerListener):
        def opened(self, event):
            logging.info("Server {0.server_address} added to topology "
                         "{0.topology_id}".format(event))
        def description_changed(self, event):
            previous_server_type = event.previous_description.server_type
            new_server_type = event.new_description.server_type
            if new_server_type != previous_server_type:
                logging.info(
                    "Server {0.server_address} changed type from "
                    "{0.previous_description.server_type_name} to "
                    "{0.new_description.server_type_name}".format(event))
        def closed(self, event):
            logging.warning("Server {0.server_address} removed from topology "
                            "{0.topology_id}".format(event))
    class HeartbeatLogger(monitoring.ServerHeartbeatListener):
        def started(self, event):
            logging.info("Heartbeat sent to server "
                         "{0.connection_id}".format(event))
        def succeeded(self, event):
            logging.info("Heartbeat to server {0.connection_id} "
                         "succeeded with reply "
                         "{0.reply.document}".format(event))
        def failed(self, event):
            logging.warning("Heartbeat to server {0.connection_id} "
                            "failed with error {0.reply}".format(event))
    class TopologyLogger(monitoring.TopologyListener):
        def opened(self, event):
            logging.info("Topology with id {0.topology_id} "
                         "opened".format(event))
        def description_changed(self, event):
            logging.info("Topology description updated for "
                         "topology id {0.topology_id}".format(event))
            previous_topology_type = event.previous_description.topology_type
            new_topology_type = event.new_description.topology_type
            if new_topology_type != previous_topology_type:
                logging.info(
                    "Topology {0.topology_id} changed type from "
                    "{0.previous_description.topology_type_name} to "
                    "{0.new_description.topology_type_name}".format(event))
            if not event.new_description.has_writable_server():
                logging.warning("No writable servers available.")
            if not event.new_description.has_readable_server():
                logging.warning("No readable servers available.")
        def closed(self, event):
            logging.info("Topology with id {0.topology_id} "
                         "closed".format(event))
Event listeners can also be registered per instance of
:class:`~pymongo.mongo_client.MongoClient`::
    client = MongoClient(event_listeners=[CommandLogger()])
Note that previously registered global listeners are automatically included
when configuring per client event listeners. Registering a new global listener
will not add that listener to existing client instances.
.. note:: Events are delivered **synchronously**. Application threads block
  waiting for event handlers (e.g. :meth:`~CommandListener.started`) to
  return. Care must be taken to ensure that your event handlers are efficient
  enough to not adversely affect overall application performance.
.. warning:: The command documents published through this API are *not* copies.
  If you intend to modify them in any way you must copy them in your event
  handler first.
"""
import sys
import traceback
from collections import namedtuple, Sequence
from pymongo.helpers import _handle_exception
_Listeners = namedtuple('Listeners',
                        ('command_listeners', 'server_listeners',
                         'server_heartbeat_listeners', 'topology_listeners'))
_LISTENERS = _Listeners([], [], [], [])
class _EventListener(object):
class CommandListener(_EventListener):
    def started(self, event):
        raise NotImplementedError
    def succeeded(self, event):
        raise NotImplementedError
    def failed(self, event):
        raise NotImplementedError
class ServerHeartbeatListener(_EventListener):
    def started(self, event):
        raise NotImplementedError
    def succeeded(self, event):
        raise NotImplementedError
    def failed(self, event):
        raise NotImplementedError
class TopologyListener(_EventListener):
    def opened(self, event):
        raise NotImplementedError
    def description_changed(self, event):
        raise NotImplementedError
    def closed(self, event):
        raise NotImplementedError
class ServerListener(_EventListener):
    def opened(self, event):
        raise NotImplementedError
    def description_changed(self, event):
        raise NotImplementedError
    def closed(self, event):
        raise NotImplementedError
def _to_micros(dur):
    if hasattr(dur, 'total_seconds'):
        return int(dur.total_seconds() * 10e5)
    return dur.microseconds + (dur.seconds + dur.days * 24 * 3600) * 1000000
def _validate_event_listeners(option, listeners):
    if not isinstance(listeners, Sequence):
        raise TypeError("%s must be a list or tuple" % (option,))
    for listener in listeners:
        if not isinstance(listener, _EventListener):
            raise TypeError("Listeners for %s must be either a "
                            "CommandListener, ServerHeartbeatListener, "
                            "ServerListener, or TopologyListener." % (option,))
    return listeners
def register(listener):
    if not isinstance(listener, _EventListener):
        raise TypeError("Listeners for %s must be either a "
                        "CommandListener, ServerHeartbeatListener, "
                        "ServerListener, or TopologyListener." % (listener,))
    if isinstance(listener, CommandListener):
        _LISTENERS.command_listeners.append(listener)
    if isinstance(listener, ServerHeartbeatListener):
        _LISTENERS.server_heartbeat_listeners.append(listener)
    if isinstance(listener, ServerListener):
        _LISTENERS.server_listeners.append(listener)
    if isinstance(listener, TopologyListener):
        _LISTENERS.topology_listeners.append(listener)
_SENSITIVE_COMMANDS = set(
    ["authenticate", "saslstart", "saslcontinue", "getnonce", "createuser",
     "updateuser", "copydbgetnonce", "copydbsaslstart", "copydb"])
class _CommandEvent(object):
    __slots__ = ("__cmd_name", "__rqst_id", "__conn_id", "__op_id")
    def __init__(self, command_name, request_id, connection_id, operation_id):
        self.__cmd_name = command_name
        self.__rqst_id = request_id
        self.__conn_id = connection_id
        self.__op_id = operation_id
    @property
    def command_name(self):
        return self.__cmd_name
    @property
    def request_id(self):
        return self.__rqst_id
    @property
    def connection_id(self):
        return self.__conn_id
    @property
    def operation_id(self):
        return self.__op_id
class CommandStartedEvent(_CommandEvent):
    __slots__ = ("__cmd", "__db")
    def __init__(self, command, database_name, *args):
        if not command:
            raise ValueError("%r is not a valid command" % (command,))
        command_name = next(iter(command))
        super(CommandStartedEvent, self).__init__(command_name, *args)
        if command_name.lower() in _SENSITIVE_COMMANDS:
            self.__cmd = {}
        else:
            self.__cmd = command
        self.__db = database_name
    @property
    def command(self):
        return self.__cmd
    @property
    def database_name(self):
        return self.__db
class CommandSucceededEvent(_CommandEvent):
    __slots__ = ("__duration_micros", "__reply")
    def __init__(self, duration, reply, command_name,
                 request_id, connection_id, operation_id):
        super(CommandSucceededEvent, self).__init__(
            command_name, request_id, connection_id, operation_id)
        self.__duration_micros = _to_micros(duration)
        if command_name.lower() in _SENSITIVE_COMMANDS:
            self.__reply = {}
        else:
            self.__reply = reply
    @property
    def duration_micros(self):
        return self.__duration_micros
    @property
    def reply(self):
        return self.__reply
class CommandFailedEvent(_CommandEvent):
    __slots__ = ("__duration_micros", "__failure")
    def __init__(self, duration, failure, *args):
        super(CommandFailedEvent, self).__init__(*args)
        self.__duration_micros = _to_micros(duration)
        self.__failure = failure
    @property
    def duration_micros(self):
        return self.__duration_micros
    @property
    def failure(self):
        return self.__failure
class _ServerEvent(object):
    __slots__ = ("__server_address", "__topology_id")
    def __init__(self, server_address, topology_id):
        self.__server_address = server_address
        self.__topology_id = topology_id
    @property
    def server_address(self):
        return self.__server_address
    @property
    def topology_id(self):
        return self.__topology_id
class ServerDescriptionChangedEvent(_ServerEvent):
    __slots__ = ('__previous_description', '__new_description')
    def __init__(self, previous_description, new_description, *args):
        super(ServerDescriptionChangedEvent, self).__init__(*args)
        self.__previous_description = previous_description
        self.__new_description = new_description
    @property
    def previous_description(self):
        return self.__previous_description
    @property
    def new_description(self):
        return self.__new_description
class ServerOpeningEvent(_ServerEvent):
    __slots__ = ()
class ServerClosedEvent(_ServerEvent):
    __slots__ = ()
class TopologyEvent(object):
    __slots__ = ('__topology_id')
    def __init__(self, topology_id):
        self.__topology_id = topology_id
    @property
    def topology_id(self):
        return self.__topology_id
class TopologyDescriptionChangedEvent(TopologyEvent):
    __slots__ = ('__previous_description', '__new_description')
    def __init__(self, previous_description,  new_description, *args):
        super(TopologyDescriptionChangedEvent, self).__init__(*args)
        self.__previous_description = previous_description
        self.__new_description = new_description
    @property
    def previous_description(self):
        return self.__previous_description
    @property
    def new_description(self):
        return self.__new_description
class TopologyOpenedEvent(TopologyEvent):
    __slots__ = ()
class TopologyClosedEvent(TopologyEvent):
    __slots__ = ()
class _ServerHeartbeatEvent(object):
    __slots__ = ('__connection_id')
    def __init__(self, connection_id):
        self.__connection_id = connection_id
    @property
    def connection_id(self):
        return self.__connection_id
class ServerHeartbeatStartedEvent(_ServerHeartbeatEvent):
    __slots__ = ()
class ServerHeartbeatSucceededEvent(_ServerHeartbeatEvent):
    __slots__ = ('__duration', '__reply')
    def __init__(self, duration, reply, *args):
        super(ServerHeartbeatSucceededEvent, self).__init__(*args)
        self.__duration = duration
        self.__reply = reply
    @property
    def duration(self):
        return self.__duration
    @property
    def reply(self):
        return self.__reply
class ServerHeartbeatFailedEvent(_ServerHeartbeatEvent):
    __slots__ = ('__duration', '__reply')
    def __init__(self, duration, reply, *args):
        super(ServerHeartbeatFailedEvent, self).__init__(*args)
        self.__duration = duration
        self.__reply = reply
    @property
    def duration(self):
        return self.__duration
    @property
    def reply(self):
        return self.__reply
class _EventListeners(object):
    def __init__(self, listeners):
        self.__command_listeners = _LISTENERS.command_listeners[:]
        self.__server_listeners = _LISTENERS.server_listeners[:]
        lst = _LISTENERS.server_heartbeat_listeners
        self.__server_heartbeat_listeners = lst[:]
        self.__topology_listeners = _LISTENERS.topology_listeners[:]
        if listeners is not None:
            for lst in listeners:
                if isinstance(lst, CommandListener):
                    self.__command_listeners.append(lst)
                if isinstance(lst, ServerListener):
                    self.__server_listeners.append(lst)
                if isinstance(lst, ServerHeartbeatListener):
                    self.__server_heartbeat_listeners.append(lst)
                if isinstance(lst, TopologyListener):
                    self.__topology_listeners.append(lst)
        self.__enabled_for_commands = bool(self.__command_listeners)
        self.__enabled_for_server = bool(self.__server_listeners)
        self.__enabled_for_server_heartbeat = bool(
            self.__server_heartbeat_listeners)
        self.__enabled_for_topology = bool(self.__topology_listeners)
    @property
    def enabled_for_commands(self):
        return self.__enabled_for_commands
    @property
    def enabled_for_server(self):
        return self.__enabled_for_server
    @property
    def enabled_for_server_heartbeat(self):
        return self.__enabled_for_server_heartbeat
    @property
    def enabled_for_topology(self):
        return self.__enabled_for_topology
    def event_listeners(self):
        return (self.__command_listeners[:],
                self.__server_heartbeat_listeners[:],
                self.__server_listeners[:],
                self.__topology_listeners[:])
    def publish_command_start(self, command, database_name,
                              request_id, connection_id, op_id=None):
        if op_id is None:
            op_id = request_id
        event = CommandStartedEvent(
            command, database_name, request_id, connection_id, op_id)
        for subscriber in self.__command_listeners:
            try:
                subscriber.started(event)
            except Exception:
                _handle_exception()
    def publish_command_success(self, duration, reply, command_name,
                                request_id, connection_id, op_id=None):
        if op_id is None:
            op_id = request_id
        event = CommandSucceededEvent(
            duration, reply, command_name, request_id, connection_id, op_id)
        for subscriber in self.__command_listeners:
            try:
                subscriber.succeeded(event)
            except Exception:
                _handle_exception()
    def publish_command_failure(self, duration, failure, command_name,
                                request_id, connection_id, op_id=None):
        if op_id is None:
            op_id = request_id
        event = CommandFailedEvent(
            duration, failure, command_name, request_id, connection_id, op_id)
        for subscriber in self.__command_listeners:
            try:
                subscriber.failed(event)
            except Exception:
                _handle_exception()
    def publish_server_heartbeat_started(self, connection_id):
        event = ServerHeartbeatStartedEvent(connection_id)
        for subscriber in self.__server_heartbeat_listeners:
            try:
                subscriber.started(event)
            except Exception:
                _handle_exception()
    def publish_server_heartbeat_succeeded(self, connection_id, duration,
                                           reply):
        event = ServerHeartbeatSucceededEvent(duration, reply, connection_id)
        for subscriber in self.__server_heartbeat_listeners:
            try:
                subscriber.succeeded(event)
            except Exception:
                _handle_exception()
    def publish_server_heartbeat_failed(self, connection_id, duration, reply):
        event = ServerHeartbeatFailedEvent(duration, reply, connection_id)
        for subscriber in self.__server_heartbeat_listeners:
            try:
                subscriber.failed(event)
            except Exception:
                _handle_exception()
    def publish_server_opened(self, server_address, topology_id):
        event = ServerOpeningEvent(server_address, topology_id)
        for subscriber in self.__server_listeners:
            try:
                subscriber.opened(event)
            except Exception:
                _handle_exception()
    def publish_server_closed(self, server_address, topology_id):
        event = ServerClosedEvent(server_address, topology_id)
        for subscriber in self.__server_listeners:
            try:
                subscriber.closed(event)
            except Exception:
                _handle_exception()
    def publish_server_description_changed(self, previous_description,
                                           new_description, server_address,
                                           topology_id):
        event = ServerDescriptionChangedEvent(previous_description,
                                              new_description, server_address,
                                              topology_id)
        for subscriber in self.__server_listeners:
            try:
                subscriber.description_changed(event)
            except Exception:
                _handle_exception()
    def publish_topology_opened(self, topology_id):
        event = TopologyOpenedEvent(topology_id)
        for subscriber in self.__topology_listeners:
            try:
                subscriber.opened(event)
            except Exception:
                _handle_exception()
    def publish_topology_closed(self, topology_id):
        event = TopologyClosedEvent(topology_id)
        for subscriber in self.__topology_listeners:
            try:
                subscriber.closed(event)
            except Exception:
                _handle_exception()
    def publish_topology_description_changed(self, previous_description,
                                             new_description, topology_id):
        event = TopologyDescriptionChangedEvent(previous_description,
                                                new_description, topology_id)
        for subscriber in self.__topology_listeners:
            try:
                subscriber.description_changed(event)
            except Exception:
                _handle_exception()
