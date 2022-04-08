
import sys
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('myflag', False, '')
def main(argv):
  if (len(argv) != 3):
    print("Length of argv was not 3: ", argv)
    sys.exit(-1)
  if argv[1] != "--passthrough":
    print("--passthrough argument not in argv")
    sys.exit(-1)
  if argv[2] != "extra":
    print("'extra' argument not in argv")
    sys.exit(-1)
if __name__ == '__main__':
  sys.argv.extend(["--myflag", "--passthrough", "extra"])
  app.run()
