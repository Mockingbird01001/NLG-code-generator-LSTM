
"""
    markupsafe._native
    ~~~~~~~~~~~~~~~~~~
    Native Python implementation the C module is not compiled.
    :copyright: (c) 2010 by Armin Ronacher.
    :license: BSD, see LICENSE for more details.
"""
from markupsafe import Markup
from markupsafe._compat import text_type
def escape(s):
    if hasattr(s, '__html__'):
        return s.__html__()
    return Markup(text_type(s)
        .replace('&', '&amp;')
        .replace('>', '&gt;')
        .replace('<', '&lt;')
    )
def escape_silent(s):
    if s is None:
        return Markup()
    return escape(s)
def soft_unicode(s):
    if not isinstance(s, text_type):
        s = text_type(s)
    return s
