from __future__ import absolute_import
from future.utils import PY3
if PY3:
    import copyreg, sys
    sys.modules['future.moves.copyreg'] = copyreg
else:
    __future_module__ = True
    from copy_reg import *
