
from absl import app
from tensorflow.tools.test import system_info_lib
def main(unused_args):
  config = system_info_lib.gather_machine_configuration()
  print(config)
if __name__ == "__main__":
