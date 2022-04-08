
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future.builtins import str
__all__ = [
    'encode_7or8bit',
    'encode_base64',
    'encode_noop',
    'encode_quopri',
    ]
try:
    from base64 import encodebytes as _bencode
except ImportError:
    from base64 import encodestring as _bencode
from quopri import encodestring as _encodestring
def _qencode(s):
    enc = _encodestring(s, quotetabs=True)
    return enc.replace(' ', '=20')
def encode_base64(msg):
    orig = msg.get_payload()
    encdata = str(_bencode(orig), 'ascii')
    msg.set_payload(encdata)
    msg['Content-Transfer-Encoding'] = 'base64'
def encode_quopri(msg):
    orig = msg.get_payload()
    encdata = _qencode(orig)
    msg.set_payload(encdata)
    msg['Content-Transfer-Encoding'] = 'quoted-printable'
def encode_7or8bit(msg):
    orig = msg.get_payload()
    if orig is None:
        msg['Content-Transfer-Encoding'] = '7bit'
        return
    try:
        if isinstance(orig, str):
            orig.encode('ascii')
        else:
            orig.decode('ascii')
    except UnicodeError:
        charset = msg.get_charset()
        output_cset = charset and charset.output_charset
        if output_cset and output_cset.lower().startswith('iso-2022-'):
            msg['Content-Transfer-Encoding'] = '7bit'
        else:
            msg['Content-Transfer-Encoding'] = '8bit'
    else:
        msg['Content-Transfer-Encoding'] = '7bit'
    if not isinstance(orig, str):
        msg.set_payload(orig.decode('ascii', 'surrogateescape'))
def encode_noop(msg):
    orig = msg.get_payload()
    if not isinstance(orig, str):
        msg.set_payload(orig.decode('ascii', 'surrogateescape'))
