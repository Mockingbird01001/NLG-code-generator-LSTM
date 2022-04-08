import os
import sys
import shutil
def get_next_cache_dir():
    work_dir = os.getcwd()
    try:
        os.chdir(os.path.dirname(sys.executable))
        import comtypes.client
        return comtypes.client._code_cache._find_gen_dir()
    except ImportError:
        return None
    finally:
        os.chdir(work_dir)
def _remove(directory):
    shutil.rmtree(directory)
    print('Removed directory "%s"' % directory)
def remove_directory(directory, silent):
    if directory:
        if silent:
            _remove(directory)
        else:
            try:
                confirm = raw_input('Remove comtypes cache directories? (y/n): ')
            except NameError:
                confirm = input('Remove comtypes cache directories? (y/n): ')
            if confirm.lower() == 'y':
                _remove(directory)
            else:
                print('Directory "%s" NOT removed' % directory)
                return False
    return True
if len(sys.argv) > 1 and "-y" in sys.argv[1:]:
    silent = True
else:
    silent = False
directory = get_next_cache_dir()
removed = remove_directory(directory, silent)
if removed:
    directory = get_next_cache_dir()
    remove_directory(directory, silent=removed)
