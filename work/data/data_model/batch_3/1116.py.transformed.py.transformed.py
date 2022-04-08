
from collections import namedtuple
from pymongo import common
from pymongo.errors import ConfigurationError
from pymongo.read_preferences import ReadPreference
from pymongo.server_description import ServerDescription
from pymongo.server_selectors import Selection
from pymongo.server_type import SERVER_TYPE
TOPOLOGY_TYPE = namedtuple('TopologyType', ['Single', 'ReplicaSetNoPrimary',
                                            'ReplicaSetWithPrimary', 'Sharded',
                                            'Unknown'])(*range(5))
class TopologyDescription(object):
    def __init__(self,
                 topology_type,
                 server_descriptions,
                 replica_set_name,
                 max_set_version,
                 max_election_id,
                 topology_settings):
        self._topology_type = topology_type
        self._replica_set_name = replica_set_name
        self._server_descriptions = server_descriptions
        self._max_set_version = max_set_version
        self._max_election_id = max_election_id
        self._topology_settings = topology_settings
        self._incompatible_err = None
        for s in self._server_descriptions.values():
            server_too_new = (
                s.min_wire_version is not None
                and s.min_wire_version > common.MAX_SUPPORTED_WIRE_VERSION)
            server_too_old = (
                s.max_wire_version is not None
                and s.max_wire_version < common.MIN_SUPPORTED_WIRE_VERSION)
            if server_too_new or server_too_old:
                self._incompatible_err = (
                    "Server at %s:%d "
                    "uses wire protocol versions %d through %d, "
                    "but PyMongo only supports %d through %d"
                    % (s.address[0], s.address[1],
                       s.min_wire_version, s.max_wire_version,
                       common.MIN_SUPPORTED_WIRE_VERSION,
                       common.MAX_SUPPORTED_WIRE_VERSION))
                break
    def check_compatible(self):
        if self._incompatible_err:
            raise ConfigurationError(self._incompatible_err)
    def has_server(self, address):
        return address in self._server_descriptions
    def reset_server(self, address):
        return updated_topology_description(self, ServerDescription(address))
    def reset(self):
        if self._topology_type == TOPOLOGY_TYPE.ReplicaSetWithPrimary:
            topology_type = TOPOLOGY_TYPE.ReplicaSetNoPrimary
        else:
            topology_type = self._topology_type
        sds = dict((address, ServerDescription(address))
                   for address in self._server_descriptions)
        return TopologyDescription(
            topology_type,
            sds,
            self._replica_set_name,
            self._max_set_version,
            self._max_election_id,
            self._topology_settings)
    def server_descriptions(self):
        return self._server_descriptions.copy()
    @property
    def topology_type(self):
        return self._topology_type
    @property
    def topology_type_name(self):
        return TOPOLOGY_TYPE._fields[self._topology_type]
    @property
    def replica_set_name(self):
        return self._replica_set_name
    @property
    def max_set_version(self):
        return self._max_set_version
    @property
    def max_election_id(self):
        return self._max_election_id
    @property
    def known_servers(self):
        return [s for s in self._server_descriptions.values()
                if s.is_server_type_known]
    @property
    def common_wire_version(self):
        servers = self.known_servers
        if servers:
            return min(s.max_wire_version for s in self.known_servers)
        return None
    @property
    def heartbeat_frequency(self):
        return self._topology_settings.heartbeat_frequency
    def apply_selector(self, selector, address):
        def apply_local_threshold(selection):
            if not selection:
                return []
            settings = self._topology_settings
            fastest = min(
                s.round_trip_time for s in selection.server_descriptions)
            threshold = settings.local_threshold_ms / 1000.0
            return [s for s in selection.server_descriptions
                    if (s.round_trip_time - fastest) <= threshold]
        if getattr(selector, 'min_wire_version', 0):
            common_wv = self.common_wire_version
            if common_wv and common_wv < selector.min_wire_version:
                raise ConfigurationError(
                    "%s requires min wire version %d, but topology's min"
                    " wire version is %d" % (selector,
                                             selector.min_wire_version,
                                             common_wv))
        if self.topology_type == TOPOLOGY_TYPE.Single:
            return self.known_servers
        elif address:
            description = self.server_descriptions().get(address)
            return [description] if description else []
        elif self.topology_type == TOPOLOGY_TYPE.Sharded:
            return apply_local_threshold(
                Selection.from_topology_description(self))
        else:
            return apply_local_threshold(
                selector(Selection.from_topology_description(self)))
    def has_readable_server(self, read_preference=ReadPreference.PRIMARY):
        common.validate_read_preference("read_preference", read_preference)
        return any(self.apply_selector(read_preference, None))
    def has_writable_server(self):
        return self.has_readable_server(ReadPreference.PRIMARY)
_SERVER_TYPE_TO_TOPOLOGY_TYPE = {
    SERVER_TYPE.Mongos: TOPOLOGY_TYPE.Sharded,
    SERVER_TYPE.RSPrimary: TOPOLOGY_TYPE.ReplicaSetWithPrimary,
    SERVER_TYPE.RSSecondary: TOPOLOGY_TYPE.ReplicaSetNoPrimary,
    SERVER_TYPE.RSArbiter: TOPOLOGY_TYPE.ReplicaSetNoPrimary,
    SERVER_TYPE.RSOther: TOPOLOGY_TYPE.ReplicaSetNoPrimary,
}
def updated_topology_description(topology_description, server_description):
    address = server_description.address
    topology_type = topology_description.topology_type
    set_name = topology_description.replica_set_name
    max_set_version = topology_description.max_set_version
    max_election_id = topology_description.max_election_id
    server_type = server_description.server_type
    sds = topology_description.server_descriptions()
    sds[address] = server_description
    if topology_type == TOPOLOGY_TYPE.Single:
        return TopologyDescription(
            TOPOLOGY_TYPE.Single,
            sds,
            set_name,
            max_set_version,
            max_election_id,
            topology_description._topology_settings)
    if topology_type == TOPOLOGY_TYPE.Unknown:
        if server_type == SERVER_TYPE.Standalone:
            sds.pop(address)
        elif server_type not in (SERVER_TYPE.Unknown, SERVER_TYPE.RSGhost):
            topology_type = _SERVER_TYPE_TO_TOPOLOGY_TYPE[server_type]
    if topology_type == TOPOLOGY_TYPE.Sharded:
        if server_type not in (SERVER_TYPE.Mongos, SERVER_TYPE.Unknown):
            sds.pop(address)
    elif topology_type == TOPOLOGY_TYPE.ReplicaSetNoPrimary:
        if server_type in (SERVER_TYPE.Standalone, SERVER_TYPE.Mongos):
            sds.pop(address)
        elif server_type == SERVER_TYPE.RSPrimary:
            (topology_type,
             set_name,
             max_set_version,
             max_election_id) = _update_rs_from_primary(sds,
                                                        set_name,
                                                        server_description,
                                                        max_set_version,
                                                        max_election_id)
        elif server_type in (
                SERVER_TYPE.RSSecondary,
                SERVER_TYPE.RSArbiter,
                SERVER_TYPE.RSOther):
            topology_type, set_name = _update_rs_no_primary_from_member(
                sds, set_name, server_description)
    elif topology_type == TOPOLOGY_TYPE.ReplicaSetWithPrimary:
        if server_type in (SERVER_TYPE.Standalone, SERVER_TYPE.Mongos):
            sds.pop(address)
            topology_type = _check_has_primary(sds)
        elif server_type == SERVER_TYPE.RSPrimary:
            (topology_type,
             set_name,
             max_set_version,
             max_election_id) = _update_rs_from_primary(sds,
                                                        set_name,
                                                        server_description,
                                                        max_set_version,
                                                        max_election_id)
        elif server_type in (
                SERVER_TYPE.RSSecondary,
                SERVER_TYPE.RSArbiter,
                SERVER_TYPE.RSOther):
            topology_type = _update_rs_with_primary_from_member(
                sds, set_name, server_description)
        else:
            topology_type = _check_has_primary(sds)
    return TopologyDescription(topology_type,
                               sds,
                               set_name,
                               max_set_version,
                               max_election_id,
                               topology_description._topology_settings)
def _update_rs_from_primary(
        sds,
        replica_set_name,
        server_description,
        max_set_version,
        max_election_id):
    if replica_set_name is None:
        replica_set_name = server_description.replica_set_name
    elif replica_set_name != server_description.replica_set_name:
        sds.pop(server_description.address)
        return (_check_has_primary(sds),
                replica_set_name,
                max_set_version,
                max_election_id)
    max_election_tuple = max_set_version, max_election_id
    if None not in server_description.election_tuple:
        if (None not in max_election_tuple and
                max_election_tuple > server_description.election_tuple):
            address = server_description.address
            sds[address] = ServerDescription(address)
            return (_check_has_primary(sds),
                    replica_set_name,
                    max_set_version,
                    max_election_id)
        max_election_id = server_description.election_id
    if (server_description.set_version is not None and
        (max_set_version is None or
            server_description.set_version > max_set_version)):
        max_set_version = server_description.set_version
    for server in sds.values():
        if (server.server_type is SERVER_TYPE.RSPrimary
                and server.address != server_description.address):
            sds[server.address] = ServerDescription(server.address)
            break
    for new_address in server_description.all_hosts:
        if new_address not in sds:
            sds[new_address] = ServerDescription(new_address)
    for addr in set(sds) - server_description.all_hosts:
        sds.pop(addr)
    return (_check_has_primary(sds),
            replica_set_name,
            max_set_version,
            max_election_id)
def _update_rs_with_primary_from_member(
        sds,
        replica_set_name,
        server_description):
    assert replica_set_name is not None
    if replica_set_name != server_description.replica_set_name:
        sds.pop(server_description.address)
    elif (server_description.me and
          server_description.address != server_description.me):
        sds.pop(server_description.address)
    return _check_has_primary(sds)
def _update_rs_no_primary_from_member(
        sds,
        replica_set_name,
        server_description):
    topology_type = TOPOLOGY_TYPE.ReplicaSetNoPrimary
    if replica_set_name is None:
        replica_set_name = server_description.replica_set_name
    elif replica_set_name != server_description.replica_set_name:
        sds.pop(server_description.address)
        return topology_type, replica_set_name
    for address in server_description.all_hosts:
        if address not in sds:
            sds[address] = ServerDescription(address)
    if (server_description.me and
            server_description.address != server_description.me):
        sds.pop(server_description.address)
    return topology_type, replica_set_name
def _check_has_primary(sds):
    for s in sds.values():
        if s.server_type == SERVER_TYPE.RSPrimary:
            return TOPOLOGY_TYPE.ReplicaSetWithPrimary
    else:
        return TOPOLOGY_TYPE.ReplicaSetNoPrimary
