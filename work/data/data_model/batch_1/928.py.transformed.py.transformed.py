from __future__ import absolute_import
from email.errors import MultipartInvariantViolationDefect, StartBoundaryNotFoundDefect
from ..exceptions import HeaderParsingError
from ..packages.six.moves import http_client as httplib
def is_fp_closed(obj):
    try:
        return obj.isclosed()
    except AttributeError:
        pass
    try:
        return obj.closed
    except AttributeError:
        pass
    try:
        return obj.fp is None
    except AttributeError:
        pass
    raise ValueError("Unable to determine whether fp is closed.")
def assert_header_parsing(headers):
    if not isinstance(headers, httplib.HTTPMessage):
        raise TypeError("expected httplib.Message, got {0}.".format(type(headers)))
    defects = getattr(headers, "defects", None)
    get_payload = getattr(headers, "get_payload", None)
    unparsed_data = None
    if get_payload:
        if not headers.is_multipart():
            payload = get_payload()
            if isinstance(payload, (bytes, str)):
                unparsed_data = payload
    if defects:
        defects = [
            defect
            for defect in defects
            if not isinstance(
                defect, (StartBoundaryNotFoundDefect, MultipartInvariantViolationDefect)
            )
        ]
    if defects or unparsed_data:
        raise HeaderParsingError(defects=defects, unparsed_data=unparsed_data)
def is_response_to_head(response):
    method = response._method
    if isinstance(method, int):
        return method == 3
    return method.upper() == "HEAD"
