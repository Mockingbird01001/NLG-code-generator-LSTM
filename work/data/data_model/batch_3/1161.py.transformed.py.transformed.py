
import re
import warnings
from bson.py3compat import PY3, iteritems, string_type
if PY3:
    from urllib.parse import unquote_plus
else:
    from urllib import unquote_plus
from pymongo.common import get_validated_options
from pymongo.errors import ConfigurationError, InvalidURI
SCHEME = 'mongodb://'
SCHEME_LEN = len(SCHEME)
DEFAULT_PORT = 27017
def _partition(entity, sep):
    parts = entity.split(sep, 1)
    if len(parts) == 2:
        return parts[0], sep, parts[1]
    else:
        return entity, '', ''
def _rpartition(entity, sep):
    idx = entity.rfind(sep)
    if idx == -1:
        return '', '', entity
    return entity[:idx], sep, entity[idx + 1:]
def parse_userinfo(userinfo):
    if '@' in userinfo or userinfo.count(':') > 1:
        if PY3:
            quote_fn = "urllib.parse.quote_plus"
        else:
            quote_fn = "urllib.quote_plus"
        raise InvalidURI("Username and password must be escaped according to "
                         "RFC 3986, use %s()." % quote_fn)
    user, _, passwd = _partition(userinfo, ":")
    if not user:
        raise InvalidURI("The empty string is not valid username.")
    return unquote_plus(user), unquote_plus(passwd)
def parse_ipv6_literal_host(entity, default_port):
    if entity.find(']') == -1:
        raise ValueError("an IPv6 address literal must be "
                         "enclosed in '[' and ']' according "
                         "to RFC 2732.")
    i = entity.find(']:')
    if i == -1:
        return entity[1:-1], default_port
    return entity[1: i], entity[i + 2:]
def parse_host(entity, default_port=DEFAULT_PORT):
    host = entity
    port = default_port
    if entity[0] == '[':
        host, port = parse_ipv6_literal_host(entity, default_port)
    elif entity.endswith(".sock"):
        return entity, default_port
    elif entity.find(':') != -1:
        if entity.count(':') > 1:
            raise ValueError("Reserved characters such as ':' must be "
                             "escaped according RFC 2396. An IPv6 "
                             "address literal must be enclosed in '[' "
                             "and ']' according to RFC 2732.")
        host, port = host.split(':', 1)
    if isinstance(port, string_type):
        if not port.isdigit() or int(port) > 65535 or int(port) <= 0:
            raise ValueError("Port must be an integer between 0 and 65535: %s"
                             % (port,))
        port = int(port)
    return host.lower(), port
def validate_options(opts, warn=False):
    return get_validated_options(opts, warn)
def _parse_options(opts, delim):
    options = {}
    for opt in opts.split(delim):
        key, val = opt.split("=")
        if key.lower() == 'readpreferencetags':
            options.setdefault('readpreferencetags', []).append(val)
        else:
            if str(key) in options:
                warnings.warn("Duplicate URI option %s" % (str(key),))
            options[str(key)] = unquote_plus(val)
    if "wtimeout" in options:
        if "wtimeoutMS" in options:
            options.pop("wtimeout")
        warnings.warn("Option wtimeout is deprecated, use 'wtimeoutMS'"
                      " instead")
    return options
def split_options(opts, validate=True, warn=False):
    and_idx = opts.find("&")
    semi_idx = opts.find(";")
    try:
        if and_idx >= 0 and semi_idx >= 0:
            raise InvalidURI("Can not mix '&' and ';' for option separators.")
        elif and_idx >= 0:
            options = _parse_options(opts, "&")
        elif semi_idx >= 0:
            options = _parse_options(opts, ";")
        elif opts.find("=") != -1:
            options = _parse_options(opts, None)
        else:
            raise ValueError
    except ValueError:
        raise InvalidURI("MongoDB URI options are key=value pairs.")
    if validate:
        return validate_options(options, warn)
    return options
def split_hosts(hosts, default_port=DEFAULT_PORT):
    nodes = []
    for entity in hosts.split(','):
        if not entity:
            raise ConfigurationError("Empty host "
                                     "(or extra comma in host list).")
        port = default_port
        if entity.endswith('.sock'):
            port = None
        nodes.append(parse_host(entity, port))
    return nodes
_BAD_DB_CHARS = re.compile('[' + re.escape(r'/ "$') + ']')
def parse_uri(uri, default_port=DEFAULT_PORT, validate=True, warn=False):
    if not uri.startswith(SCHEME):
        raise InvalidURI("Invalid URI scheme: URI "
                         "must begin with '%s'" % (SCHEME,))
    scheme_free = uri[SCHEME_LEN:]
    if not scheme_free:
        raise InvalidURI("Must provide at least one hostname or IP.")
    user = None
    passwd = None
    dbase = None
    collection = None
    options = {}
    host_part, _, path_part = _partition(scheme_free, '/')
    if not host_part:
        host_part = path_part
        path_part = ""
    if not path_part and '?' in host_part:
        raise InvalidURI("A '/' is required between "
                         "the host list and any options.")
    if '@' in host_part:
        userinfo, _, hosts = _rpartition(host_part, '@')
        user, passwd = parse_userinfo(userinfo)
    else:
        hosts = host_part
    if '/' in hosts:
        raise InvalidURI("Any '/' in a unix domain socket must be"
                         " percent-encoded: %s" % host_part)
    hosts = unquote_plus(hosts)
    nodes = split_hosts(hosts, default_port=default_port)
    if path_part:
        if path_part[0] == '?':
            opts = unquote_plus(path_part[1:])
        else:
            dbase, _, opts = map(unquote_plus, _partition(path_part, '?'))
            if '.' in dbase:
                dbase, collection = dbase.split('.', 1)
            if _BAD_DB_CHARS.search(dbase):
                raise InvalidURI('Bad database name "%s"' % dbase)
        if opts:
            options = split_options(opts, validate, warn)
    if dbase is not None:
        dbase = unquote_plus(dbase)
    if collection is not None:
        collection = unquote_plus(collection)
    return {
        'nodelist': nodes,
        'username': user,
        'password': passwd,
        'database': dbase,
        'collection': collection,
        'options': options
    }
if __name__ == '__main__':
    import pprint
    import sys
    try:
        pprint.pprint(parse_uri(sys.argv[1]))
    except InvalidURI as e:
        print(e)
    sys.exit(0)
