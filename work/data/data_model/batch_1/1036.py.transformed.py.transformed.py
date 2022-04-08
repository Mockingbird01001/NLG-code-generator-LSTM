
import re
import sys
try:
    from ipaddress import ip_address
except ImportError:
    ip_address = lambda address: None
if sys.version_info[0] < 3:
    _unicode = unicode
else:
    _unicode = lambda value: value
class CertificateError(ValueError):
    pass
def _dnsname_match(dn, hostname, max_wildcards=1):
    pats = []
    if not dn:
        return False
    parts = dn.split(r'.')
    leftmost = parts[0]
    remainder = parts[1:]
    wildcards = leftmost.count('*')
    if wildcards > max_wildcards:
        raise CertificateError(
            "too many wildcards in certificate DNS name: " + repr(dn))
    if not wildcards:
        return dn.lower() == hostname.lower()
    if leftmost == '*':
        pats.append('[^.]+')
    elif leftmost.startswith('xn--') or hostname.startswith('xn--'):
        pats.append(re.escape(leftmost))
    else:
        pats.append(re.escape(leftmost).replace(r'\*', '[^.]*'))
    for frag in remainder:
        pats.append(re.escape(frag))
    pat = re.compile(r'\A' + r'\.'.join(pats) + r'\Z', re.IGNORECASE)
    return pat.match(hostname)
def _ipaddress_match(ipname, host_ip):
    ip = ip_address(_unicode(ipname).rstrip())
    return ip == host_ip
def match_hostname(cert, hostname):
    if not cert:
        raise ValueError("empty or no certificate, match_hostname needs a "
                         "SSL socket or SSL context with either "
                         "CERT_OPTIONAL or CERT_REQUIRED")
    try:
        host_ip = ip_address(_unicode(hostname))
    except (ValueError, UnicodeError):
        host_ip = None
    dnsnames = []
    san = cert.get('subjectAltName', ())
    for key, value in san:
        if key == 'DNS':
            if host_ip is None and _dnsname_match(value, hostname):
                return
            dnsnames.append(value)
        elif key == 'IP Address':
            if host_ip is not None and _ipaddress_match(value, host_ip):
                return
            dnsnames.append(value)
    if not dnsnames:
        for sub in cert.get('subject', ()):
            for key, value in sub:
                if key == 'commonName':
                    if _dnsname_match(value, hostname):
                        return
                    dnsnames.append(value)
    if len(dnsnames) > 1:
        raise CertificateError("hostname %r "
            "doesn't match either of %s"
            % (hostname, ', '.join(map(repr, dnsnames))))
    elif len(dnsnames) == 1:
        raise CertificateError("hostname %r "
            "doesn't match %r"
            % (hostname, dnsnames[0]))
    else:
        raise CertificateError("no appropriate commonName or "
            "subjectAltName fields were found")
