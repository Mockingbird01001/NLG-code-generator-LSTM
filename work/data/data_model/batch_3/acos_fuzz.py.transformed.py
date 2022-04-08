
import atheris
with atheris.instrument_imports():
  import sys
  from python_fuzzing import FuzzingHelper
  import tensorflow as tf
def TestOneInput(data):
  fh = FuzzingHelper(data)
  input_tensor = fh.get_random_numeric_tensor()
  _ = tf.raw_ops.Acos(x=input_tensor)
def main():
  atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
  atheris.Fuzz()
if __name__ == "__main__":
  main()
