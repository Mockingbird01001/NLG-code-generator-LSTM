
from pip._vendor.tenacity import _utils
def before_nothing(retry_state):
def before_log(logger, log_level):
    def log_it(retry_state):
        logger.log(
            log_level,
            "Starting call to '%s', this is the %s time calling it.",
            _utils.get_callback_name(retry_state.fn),
            _utils.to_ordinal(retry_state.attempt_number),
        )
    return log_it
