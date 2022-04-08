
__version__ = "$Revision: 1.3 $"[10:-1]
f2py_version = 'See `f2py -v`'
from .auxfuncs import (
    applyrules, dictappend, gentitle, hasnote, outmess
)
usemodule_rules = {
    'body': """
Arguments:\\n\\
\tif (!PyArg_ParseTuple(capi_args, \"\")) goto capi_fail;
\treturn Py_BuildValue(\"\");
capi_fail:
\treturn NULL;
}
""",
    'need': ['F_MODFUNC']
}
def buildusevars(m, r):
    ret = {}
    outmess(
        '\t\tBuilding use variable hooks for module "%s" (feature only for F90/F95)...\n' % (m['name']))
    varsmap = {}
    revmap = {}
    if 'map' in r:
        for k in r['map'].keys():
            if r['map'][k] in revmap:
                outmess('\t\t\tVariable "%s<=%s" is already mapped by "%s". Skipping.\n' % (
                    r['map'][k], k, revmap[r['map'][k]]))
            else:
                revmap[r['map'][k]] = k
    if 'only' in r and r['only']:
        for v in r['map'].keys():
            if r['map'][v] in m['vars']:
                if revmap[r['map'][v]] == v:
                    varsmap[v] = r['map'][v]
                else:
                    outmess('\t\t\tIgnoring map "%s=>%s". See above.\n' %
                            (v, r['map'][v]))
            else:
                outmess(
                    '\t\t\tNo definition for variable "%s=>%s". Skipping.\n' % (v, r['map'][v]))
    else:
        for v in m['vars'].keys():
            if v in revmap:
                varsmap[v] = revmap[v]
            else:
                varsmap[v] = v
    for v in varsmap.keys():
        ret = dictappend(ret, buildusevar(v, varsmap[v], m['vars'], m['name']))
    return ret
def buildusevar(name, realname, vars, usemodulename):
    outmess('\t\t\tConstructing wrapper function for variable "%s=>%s"...\n' % (
        name, realname))
    ret = {}
    vrd = {'name': name,
           'realname': realname,
           'REALNAME': realname.upper(),
           'usemodulename': usemodulename,
           'USEMODULENAME': usemodulename.upper(),
           'texname': name.replace('_', '\\_'),
           'begintitle': gentitle('%s=>%s' % (name, realname)),
           'endtitle': gentitle('end of %s=>%s' % (name, realname)),
           }
    nummap = {0: 'Ro', 1: 'Ri', 2: 'Rii', 3: 'Riii', 4: 'Riv',
              5: 'Rv', 6: 'Rvi', 7: 'Rvii', 8: 'Rviii', 9: 'Rix'}
    vrd['texnamename'] = name
    for i in nummap.keys():
        vrd['texnamename'] = vrd['texnamename'].replace(repr(i), nummap[i])
    if hasnote(vars[realname]):
        vrd['note'] = vars[realname]['note']
    rd = dictappend({}, vrd)
    print(name, realname, vars[realname])
    ret = applyrules(usemodule_rules, rd)
    return ret
