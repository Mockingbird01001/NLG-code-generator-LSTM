
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
ERROR_MESSAGE =
def main():
  if os.path.split(BASE_DIR)[-1] != 'tensorflow':
    raise AssertionError(
        "BASE_DIR = '%s' doesn't end with tensorflow" % BASE_DIR)
  for dirpath, dirnames, filenames in os.walk(BASE_DIR, followlinks=True):
    lowercase_directories = [x.lower() for x in dirnames]
    lowercase_files = [x.lower() for x in filenames]
    lowercase_dir_contents = lowercase_directories + lowercase_files
    if len(lowercase_dir_contents) != len(set(lowercase_dir_contents)):
      raise AssertionError(ERROR_MESSAGE.format(dirpath))
if __name__ == '__main__':
  main()
