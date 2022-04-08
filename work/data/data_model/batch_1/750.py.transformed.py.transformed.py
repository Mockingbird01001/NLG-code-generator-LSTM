
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
__all__ = ['MIMENonMultipart']
from future.backports.email import errors
from future.backports.email.mime.base import MIMEBase
class MIMENonMultipart(MIMEBase):
    def attach(self, payload):
        raise errors.MultipartConversionError(
            'Cannot attach additional subparts to non-multipart/*')
