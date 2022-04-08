
import atheris
with atheris.instrument_imports():
  import sys
  import tensorflow as tf
def TestOneInput(data):
  tf.constant(data)
def main():
  atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
  atheris.Fuzz()
if __name__ == "__main__":
  main()
