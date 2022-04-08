
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
__all__ = ['MIMEText']
from future.backports.email.encoders import encode_7or8bit
from future.backports.email.mime.nonmultipart import MIMENonMultipart
class MIMEText(MIMENonMultipart):
    def __init__(self, _text, _subtype='plain', _charset=None):
        if _charset is None:
            try:
                _text.encode('us-ascii')
                _charset = 'us-ascii'
            except UnicodeEncodeError:
                _charset = 'utf-8'
        MIMENonMultipart.__init__(self, 'text', _subtype,
                                  **{'charset': _charset})
        self.set_payload(_text, _charset)
