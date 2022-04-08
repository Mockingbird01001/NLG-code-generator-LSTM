from __future__ import absolute_import
import email.utils
import mimetypes
import re
from .packages import six
def guess_content_type(filename, default="application/octet-stream"):
    if filename:
        return mimetypes.guess_type(filename)[0] or default
    return default
def format_header_param_rfc2231(name, value):
    if isinstance(value, six.binary_type):
        value = value.decode("utf-8")
    if not any(ch in value for ch in '"\\\r\n'):
        result = u'%s="%s"' % (name, value)
        try:
            result.encode("ascii")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        else:
            return result
    if six.PY2:
        value = value.encode("utf-8")
    value = email.utils.encode_rfc2231(value, "utf-8")
    value = "%s*=%s" % (name, value)
    if six.PY2:
        value = value.decode("utf-8")
    return value
_HTML5_REPLACEMENTS = {
    u"\u0022": u"%22",
    u"\u005C": u"\u005C\u005C",
}
_HTML5_REPLACEMENTS.update(
    {
        six.unichr(cc): u"%{:02X}".format(cc)
        for cc in range(0x00, 0x1F + 1)
        if cc not in (0x1B,)
    }
)
def _replace_multiple(value, needles_and_replacements):
    def replacer(match):
        return needles_and_replacements[match.group(0)]
    pattern = re.compile(
        r"|".join([re.escape(needle) for needle in needles_and_replacements.keys()])
    )
    result = pattern.sub(replacer, value)
    return result
def format_header_param_html5(name, value):
    if isinstance(value, six.binary_type):
        value = value.decode("utf-8")
    value = _replace_multiple(value, _HTML5_REPLACEMENTS)
    return u'%s="%s"' % (name, value)
format_header_param = format_header_param_html5
class RequestField(object):
    def __init__(
        self,
        name,
        data,
        filename=None,
        headers=None,
        header_formatter=format_header_param_html5,
    ):
        self._name = name
        self._filename = filename
        self.data = data
        self.headers = {}
        if headers:
            self.headers = dict(headers)
        self.header_formatter = header_formatter
    @classmethod
    def from_tuples(cls, fieldname, value, header_formatter=format_header_param_html5):
        if isinstance(value, tuple):
            if len(value) == 3:
                filename, data, content_type = value
            else:
                filename, data = value
                content_type = guess_content_type(filename)
        else:
            filename = None
            content_type = None
            data = value
        request_param = cls(
            fieldname, data, filename=filename, header_formatter=header_formatter
        )
        request_param.make_multipart(content_type=content_type)
        return request_param
    def _render_part(self, name, value):
        return self.header_formatter(name, value)
    def _render_parts(self, header_parts):
        parts = []
        iterable = header_parts
        if isinstance(header_parts, dict):
            iterable = header_parts.items()
        for name, value in iterable:
            if value is not None:
                parts.append(self._render_part(name, value))
        return u"; ".join(parts)
    def render_headers(self):
        lines = []
        sort_keys = ["Content-Disposition", "Content-Type", "Content-Location"]
        for sort_key in sort_keys:
            if self.headers.get(sort_key, False):
                lines.append(u"%s: %s" % (sort_key, self.headers[sort_key]))
        for header_name, header_value in self.headers.items():
            if header_name not in sort_keys:
                if header_value:
                    lines.append(u"%s: %s" % (header_name, header_value))
        lines.append(u"\r\n")
        return u"\r\n".join(lines)
    def make_multipart(
        self, content_disposition=None, content_type=None, content_location=None
    ):
        self.headers["Content-Disposition"] = content_disposition or u"form-data"
        self.headers["Content-Disposition"] += u"; ".join(
            [
                u"",
                self._render_parts(
                    ((u"name", self._name), (u"filename", self._filename))
                ),
            ]
        )
        self.headers["Content-Type"] = content_type
        self.headers["Content-Location"] = content_location
