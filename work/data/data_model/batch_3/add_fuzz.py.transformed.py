
import atheris
with atheris.instrument_imports():
  import sys
  from python_fuzzing import FuzzingHelper
  import tensorflow as tf
def TestOneInput(data):
  fh = FuzzingHelper(data)
  input_tensor_x = fh.get_random_numeric_tensor()
  input_tensor_y = fh.get_random_numeric_tensor()
  try:
    _ = tf.raw_ops.Add(x=input_tensor_x, y=input_tensor_y)
  except (tf.errors.InvalidArgumentError, tf.errors.UnimplementedError):
    pass
def main():
  atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
  atheris.Fuzz()
if __name__ == "__main__":
  main()
