
import textwrap
def check_inline(cmd):
    cmd._check_compiler()
    body = textwrap.dedent("""
        static %(inline)s int static_func (void)
        {
            return 0;
        }
        %(inline)s int nostatic_func (void)
        {
            return 0;
        }
    for kw in ['inline', '__inline__', '__inline']:
        st = cmd.try_compile(body % {'inline': kw}, None, None)
        if st:
            return kw
    return ''
def check_restrict(cmd):
    cmd._check_compiler()
    body = textwrap.dedent("""
        static int static_func (char * %(restrict)s a)
        {
            return 0;
        }
        """)
    for kw in ['restrict', '__restrict__', '__restrict']:
        st = cmd.try_compile(body % {'restrict': kw}, None, None)
        if st:
            return kw
    return ''
def check_compiler_gcc(cmd):
    cmd._check_compiler()
    body = textwrap.dedent("""
        int
        main()
        {
            return 0;
        }
        """)
    return cmd.try_compile(body, None, None)
def check_gcc_version_at_least(cmd, major, minor=0, patchlevel=0):
    cmd._check_compiler()
    version = '.'.join([str(major), str(minor), str(patchlevel)])
    body = textwrap.dedent("""
        int
        main()
        {
                (__GNUC_MINOR__ < %(minor)d) || \\
                (__GNUC_PATCHLEVEL__ < %(patchlevel)d)
            return 0;
        }
        """)
    kw = {'version': version, 'major': major, 'minor': minor,
          'patchlevel': patchlevel}
    return cmd.try_compile(body % kw, None, None)
def check_gcc_function_attribute(cmd, attribute, name):
    cmd._check_compiler()
    body = textwrap.dedent("""
        int %s %s(void* unused)
        {
            return 0;
        }
        int
        main()
        {
            return 0;
        }
        """) % (attribute, name)
    return cmd.try_compile(body, None, None) != 0
def check_gcc_function_attribute_with_intrinsics(cmd, attribute, name, code,
                                                include):
    cmd._check_compiler()
    body = textwrap.dedent("""
        int %s %s(void)
        {
            %s;
            return 0;
        }
        int
        main()
        {
            return 0;
        }
        """) % (include, attribute, name, code)
    return cmd.try_compile(body, None, None) != 0
def check_gcc_variable_attribute(cmd, attribute):
    cmd._check_compiler()
    body = textwrap.dedent("""
        int %s foo;
        int
        main()
        {
            return 0;
        }
        """) % (attribute, )
    return cmd.try_compile(body, None, None) != 0
