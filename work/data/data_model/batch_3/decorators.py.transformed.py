
import functools
def wrapping_decorator(f):
  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    return f(*args, **kwargs)
  return wrapper
def standalone_decorator(f):
  def standalone_wrapper(*args, **kwargs):
    return f(*args, **kwargs)
  return standalone_wrapper
def functional_decorator():
  def decorator(f):
    def functional_wrapper(*args, **kwargs):
      return f(*args, **kwargs)
    return functional_wrapper
  return decorator
