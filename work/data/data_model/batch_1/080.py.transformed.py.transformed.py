
from pip._vendor.tenacity import _utils
from pip._vendor.tenacity.compat import get_exc_info_from_future
def before_sleep_nothing(retry_state):
def before_sleep_log(logger, log_level, exc_info=False):
    def log_it(retry_state):
        if retry_state.outcome.failed:
            ex = retry_state.outcome.exception()
            verb, value = "raised", "%s: %s" % (type(ex).__name__, ex)
            if exc_info:
                local_exc_info = get_exc_info_from_future(retry_state.outcome)
            else:
                local_exc_info = False
        else:
            verb, value = "returned", retry_state.outcome.result()
            local_exc_info = False
        logger.log(
            log_level,
            "Retrying %s in %s seconds as it %s %s.",
            _utils.get_callback_name(retry_state.fn),
            getattr(retry_state.next_action, "sleep"),
            verb,
            value,
            exc_info=local_exc_info,
        )
    return log_it
