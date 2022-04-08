
from collections import namedtuple
SERVER_TYPE = namedtuple('ServerType',
                         ['Unknown', 'Mongos', 'RSPrimary', 'RSSecondary',
                          'RSArbiter', 'RSOther', 'RSGhost',
                          'Standalone'])(*range(8))
