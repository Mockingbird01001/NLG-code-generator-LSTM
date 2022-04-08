
import sys
def inject_securetransport():
    if sys.platform != "darwin":
        return
    try:
        import ssl
    except ImportError:
        return
    if ssl.OPENSSL_VERSION_NUMBER >= 0x1000100F:
        return
    try:
        from pip._vendor.urllib3.contrib import securetransport
    except (ImportError, OSError):
        return
    securetransport.inject_into_urllib3()
inject_securetransport()
