
def simple_function(x):
def nested_functions(x):
  def inner_fn(y):
    return y
  return inner_fn(x)
def function_with_print():
  print('foo')
simple_lambda = lambda: None
class SimpleClass(object):
  def simple_method(self):
    return self
  def method_with_print(self):
    print('foo')
def function_with_multiline_call(x):
  return range(
      x,
      x + 1,
  )
def basic_decorator(f):
  return f
@basic_decorator
@basic_decorator
def decorated_function(x):
  if x > 0:
    return 1
  return 2
