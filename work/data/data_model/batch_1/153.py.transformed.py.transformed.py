
from bson.codec_options import _parse_codec_options
from pymongo.auth import _build_credentials_tuple
from pymongo.common import validate_boolean
from pymongo import common
from pymongo.errors import ConfigurationError
from pymongo.monitoring import _EventListeners
from pymongo.pool import PoolOptions
from pymongo.read_concern import ReadConcern
from pymongo.read_preferences import (make_read_preference,
                                      read_pref_mode_from_name)
from pymongo.ssl_support import get_ssl_context
from pymongo.write_concern import WriteConcern
def _parse_credentials(username, password, database, options):
    mechanism = options.get('authmechanism', 'DEFAULT')
    if username is None and mechanism != 'MONGODB-X509':
        return None
    source = options.get('authsource', database or 'admin')
    return _build_credentials_tuple(
        mechanism, source, username, password, options)
def _parse_read_preference(options):
    if 'read_preference' in options:
        return options['read_preference']
    name = options.get('readpreference', 'primary')
    mode = read_pref_mode_from_name(name)
    tags = options.get('readpreferencetags')
    max_staleness = options.get('maxstalenessseconds', -1)
    return make_read_preference(mode, tags, max_staleness)
def _parse_write_concern(options):
    concern = options.get('w')
    wtimeout = options.get('wtimeout')
    j = options.get('j', options.get('journal'))
    fsync = options.get('fsync')
    return WriteConcern(concern, wtimeout, j, fsync)
def _parse_read_concern(options):
    concern = options.get('readconcernlevel')
    return ReadConcern(concern)
def _parse_ssl_options(options):
    use_ssl = options.get('ssl')
    if use_ssl is not None:
        validate_boolean('ssl', use_ssl)
    certfile = options.get('ssl_certfile')
    keyfile = options.get('ssl_keyfile')
    passphrase = options.get('ssl_pem_passphrase')
    ca_certs = options.get('ssl_ca_certs')
    cert_reqs = options.get('ssl_cert_reqs')
    match_hostname = options.get('ssl_match_hostname', True)
    crlfile = options.get('ssl_crlfile')
    ssl_kwarg_keys = [k for k in options
                      if k.startswith('ssl_') and options[k]]
    if use_ssl == False and ssl_kwarg_keys:
        raise ConfigurationError("ssl has not been enabled but the "
                                 "following ssl parameters have been set: "
                                 "%s. Please set `ssl=True` or remove."
                                 % ', '.join(ssl_kwarg_keys))
    if ssl_kwarg_keys and use_ssl is None:
        use_ssl = True
    if use_ssl is True:
        ctx = get_ssl_context(
            certfile, keyfile, passphrase, ca_certs, cert_reqs, crlfile)
        return ctx, match_hostname
    return None, match_hostname
def _parse_pool_options(options):
    max_pool_size = options.get('maxpoolsize', common.MAX_POOL_SIZE)
    min_pool_size = options.get('minpoolsize', common.MIN_POOL_SIZE)
    max_idle_time_ms = options.get('maxidletimems', common.MAX_IDLE_TIME_MS)
    if max_pool_size is not None and min_pool_size > max_pool_size:
        raise ValueError("minPoolSize must be smaller or equal to maxPoolSize")
    connect_timeout = options.get('connecttimeoutms', common.CONNECT_TIMEOUT)
    socket_keepalive = options.get('socketkeepalive', True)
    socket_timeout = options.get('sockettimeoutms')
    wait_queue_timeout = options.get('waitqueuetimeoutms')
    wait_queue_multiple = options.get('waitqueuemultiple')
    event_listeners = options.get('event_listeners')
    appname = options.get('appname')
    ssl_context, ssl_match_hostname = _parse_ssl_options(options)
    return PoolOptions(max_pool_size,
                       min_pool_size,
                       max_idle_time_ms,
                       connect_timeout, socket_timeout,
                       wait_queue_timeout, wait_queue_multiple,
                       ssl_context, ssl_match_hostname, socket_keepalive,
                       _EventListeners(event_listeners),
                       appname)
class ClientOptions(object):
    def __init__(self, username, password, database, options):
        self.__options = options
        self.__codec_options = _parse_codec_options(options)
        self.__credentials = _parse_credentials(
            username, password, database, options)
        self.__local_threshold_ms = options.get(
            'localthresholdms', common.LOCAL_THRESHOLD_MS)
        self.__server_selection_timeout = options.get(
            'serverselectiontimeoutms', common.SERVER_SELECTION_TIMEOUT)
        self.__pool_options = _parse_pool_options(options)
        self.__read_preference = _parse_read_preference(options)
        self.__replica_set_name = options.get('replicaset')
        self.__write_concern = _parse_write_concern(options)
        self.__read_concern = _parse_read_concern(options)
        self.__connect = options.get('connect')
        self.__heartbeat_frequency = options.get(
            'heartbeatfrequencyms', common.HEARTBEAT_FREQUENCY)
    @property
    def _options(self):
        return self.__options
    @property
    def connect(self):
        return self.__connect
    @property
    def codec_options(self):
        return self.__codec_options
    @property
    def credentials(self):
        return self.__credentials
    @property
    def local_threshold_ms(self):
        return self.__local_threshold_ms
    @property
    def server_selection_timeout(self):
        return self.__server_selection_timeout
    @property
    def heartbeat_frequency(self):
        return self.__heartbeat_frequency
    @property
    def pool_options(self):
        return self.__pool_options
    @property
    def read_preference(self):
        return self.__read_preference
    @property
    def replica_set_name(self):
        return self.__replica_set_name
    @property
    def write_concern(self):
        return self.__write_concern
    @property
    def read_concern(self):
        return self.__read_concern
