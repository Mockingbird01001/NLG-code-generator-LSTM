
import tokenize
import sys
def run():
    __builtins__
    script_name = sys.argv[1]
    namespace = dict(
        __file__=script_name,
        __name__='__main__',
        __doc__=None,
    )
    sys.argv[:] = sys.argv[1:]
    open_ = getattr(tokenize, 'open', open)
    with open_(script_name) as fid:
        script = fid.read()
    norm_script = script.replace('\\r\\n', '\\n')
    code = compile(norm_script, script_name, 'exec')
    exec(code, namespace)
if __name__ == '__main__':
    run()
