
import functools
from tensorflow.python.util import decorator_utils
def keyword_args_only(func):
  decorator_utils.validate_callable(func, "keyword_args_only")
  @functools.wraps(func)
  def new_func(*args, **kwargs):
    if args:
      raise ValueError(
          f"The function {func.__name__} only accepts keyword arguments. "
          "Do not pass positional arguments. Received the following positional "
          f"arguments: {args}")
    return func(**kwargs)
  return new_func
