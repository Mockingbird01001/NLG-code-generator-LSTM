
"""Criteria to select ServerDescriptions based on maxStalenessSeconds.
The Max Staleness Spec says: When there is a known primary P,
a secondary S's staleness is estimated with this formula:
  (S.lastUpdateTime - S.lastWriteDate) - (P.lastUpdateTime - P.lastWriteDate)
  + heartbeatFrequencyMS
When there is no known primary, a secondary S's staleness is estimated with:
  SMax.lastWriteDate - S.lastWriteDate + heartbeatFrequencyMS
where "SMax" is the secondary with the greatest lastWriteDate.
"""
from pymongo.errors import ConfigurationError
from pymongo.server_type import SERVER_TYPE
IDLE_WRITE_PERIOD = 10
SMALLEST_MAX_STALENESS = 90
def _validate_max_staleness(max_staleness,
                            heartbeat_frequency):
    if max_staleness < heartbeat_frequency + IDLE_WRITE_PERIOD:
        raise ConfigurationError(
            "maxStalenessSeconds must be at least heartbeatFrequencyMS +"
            " %d seconds. maxStalenessSeconds is set to %d,"
            " heartbeatFrequencyMS is set to %d." % (
                IDLE_WRITE_PERIOD, max_staleness, heartbeat_frequency * 1000))
    if max_staleness < SMALLEST_MAX_STALENESS:
        raise ConfigurationError(
            "maxStalenessSeconds must be at least %d. "
            "maxStalenessSeconds is set to %d." % (
                SMALLEST_MAX_STALENESS, max_staleness))
def _with_primary(max_staleness, selection):
    primary = selection.primary
    sds = []
    for s in selection.server_descriptions:
        if s.server_type == SERVER_TYPE.RSSecondary:
            staleness = (
                (s.last_update_time - s.last_write_date) -
                (primary.last_update_time - primary.last_write_date) +
                selection.heartbeat_frequency)
            if staleness <= max_staleness:
                sds.append(s)
        else:
            sds.append(s)
    return selection.with_server_descriptions(sds)
def _no_primary(max_staleness, selection):
    smax = selection.secondary_with_max_last_write_date()
    if not smax:
        return selection.with_server_descriptions([])
    sds = []
    for s in selection.server_descriptions:
        if s.server_type == SERVER_TYPE.RSSecondary:
            staleness = (smax.last_write_date -
                         s.last_write_date +
                         selection.heartbeat_frequency)
            if staleness <= max_staleness:
                sds.append(s)
        else:
            sds.append(s)
    return selection.with_server_descriptions(sds)
def select(max_staleness, selection):
    if max_staleness == -1:
        return selection
    _validate_max_staleness(max_staleness, selection.heartbeat_frequency)
    if selection.primary:
        return _with_primary(max_staleness, selection)
    else:
        return _no_primary(max_staleness, selection)
