
import os
def path_to_string(path):
  if isinstance(path, os.PathLike):
    return os.fspath(path)
  return path
def ask_to_proceed_with_overwrite(filepath):
  overwrite = input('[WARNING] %s already exists - overwrite? '
                    '[y/n]' % (filepath)).strip().lower()
  while overwrite not in ('y', 'n'):
    overwrite = input('Enter "y" (overwrite) or "n" '
                      '(cancel).').strip().lower()
  if overwrite == 'n':
    return False
  print('[TIP] Next time specify overwrite=True!')
  return True
