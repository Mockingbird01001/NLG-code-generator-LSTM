
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
__all__ = ['MIMEMessage']
from future.backports.email import message
from future.backports.email.mime.nonmultipart import MIMENonMultipart
class MIMEMessage(MIMENonMultipart):
    def __init__(self, _msg, _subtype='rfc822'):
        MIMENonMultipart.__init__(self, 'message', _subtype)
        if not isinstance(_msg, message.Message):
            raise TypeError('Argument is not an instance of Message')
        message.Message.attach(self, _msg)
        self.set_default_type('message/rfc822')
