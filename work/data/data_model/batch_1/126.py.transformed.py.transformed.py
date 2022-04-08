
from . import __version__
from .auxfuncs import (
    applyrules, debugcapi, dictappend, errmess, getargs, hasnote, isarray,
    iscomplex, iscomplexarray, iscomplexfunction, isfunction, isintent_c,
    isintent_hide, isintent_in, isintent_inout, isintent_nothide,
    isintent_out, isoptional, isrequired, isscalar, isstring,
    isstringfunction, issubroutine, l_and, l_not, l_or, outmess, replace,
    stripcomma, throw_error
)
from . import cfuncs
f2py_version = __version__.version
cb_routine_rules = {
    'body': """
typedef struct {
    PyObject *capi;
    PyTupleObject *args_capi;
    int nofargs;
    jmp_buf jmpbuf;
    return prev;
}
}
}
}
    PyTupleObject *capi_arglist = NULL;
    PyObject *capi_return = NULL;
    PyObject *capi_tmp = NULL;
    PyObject *capi_arglist_list = NULL;
    int capi_j,capi_i = 0;
    int capi_longjmp_ok = 1;
f2py_cb_start_clock();
    if (cb == NULL) {
        capi_longjmp_ok = 0;
        cb = &cb_local;
    }
    capi_arglist = cb->args_capi;
    if (cb->capi==NULL) {
        capi_longjmp_ok = 0;
    }
    if (cb->capi==NULL) {
        goto capi_fail;
    }
    if (F2PyCapsule_Check(cb->capi)) {
    }
    if (capi_arglist==NULL) {
        capi_longjmp_ok = 0;
        if (capi_tmp) {
            capi_arglist = (PyTupleObject *)PySequence_Tuple(capi_tmp);
            if (capi_arglist==NULL) {
                goto capi_fail;
            }
        } else {
            PyErr_Clear();
            capi_arglist = (PyTupleObject *)Py_BuildValue(\"()\");
        }
    }
    if (capi_arglist == NULL) {
        goto capi_fail;
    }
    capi_arglist_list = PySequence_List(capi_arglist);
    if (capi_arglist_list == NULL) goto capi_fail;
    CFUNCSMESSPY(\"cb:capi_arglist=\",capi_arglist_list);
    CFUNCSMESSPY(\"cb:capi_arglist=\",capi_arglist);
f2py_cb_start_call_clock();
    capi_return = PyObject_CallObject(cb->capi,(PyObject *)capi_arglist_list);
    Py_DECREF(capi_arglist_list);
    capi_arglist_list = NULL;
    capi_return = PyObject_CallObject(cb->capi,(PyObject *)capi_arglist);
f2py_cb_stop_call_clock();
    CFUNCSMESSPY(\"cb:capi_return=\",capi_return);
    if (capi_return == NULL) {
        fprintf(stderr,\"capi_return is NULL\\n\");
        goto capi_fail;
    }
    if (capi_return == Py_None) {
        Py_DECREF(capi_return);
        capi_return = Py_BuildValue(\"()\");
    }
    else if (!PyTuple_Check(capi_return)) {
        capi_return = Py_BuildValue(\"(N)\",capi_return);
    }
    capi_j = PyTuple_Size(capi_return);
    capi_i = 0;
    Py_DECREF(capi_return);
f2py_cb_stop_clock();
    goto capi_return_pt;
capi_fail:
    Py_XDECREF(capi_return);
    Py_XDECREF(capi_arglist_list);
    if (capi_longjmp_ok) {
        longjmp(cb->jmpbuf,-1);
    }
capi_return_pt:
    ;
}
}
cb_rout_rules = [
    {
        'separatorsfor': {'decl': '\n',
                          'args': ',', 'optargs': '', 'pyobjfrom': '\n', 'freemem': '\n',
                          'args_td': ',', 'optargs_td': '',
                          'args_nm': ',', 'optargs_nm': '',
                          'frompyobj': '\n', 'setdims': '\n',
                          'docstrsigns': '\\n"\n"',
                          'latexdocstrsigns': '\n',
                          'latexdocstrreq': '\n', 'latexdocstropt': '\n',
                          'latexdocstrout': '\n', 'latexdocstrcbs': '\n',
                          },
        'decl': '/*decl*/', 'pyobjfrom': '/*pyobjfrom*/', 'frompyobj': '/*frompyobj*/',
        'args': [], 'optargs': '', 'return': '', 'strarglens': '', 'freemem': '/*freemem*/',
        'args_td': [], 'optargs_td': '', 'strarglens_td': '',
        'args_nm': [], 'optargs_nm': '', 'strarglens_nm': '',
        'noargs': '',
        'setdims': '/*setdims*/',
        'docstrsigns': '', 'latexdocstrsigns': '',
        'docstrreq': '\tRequired arguments:',
        'docstropt': '\tOptional arguments:',
        'docstrout': '\tReturn objects:',
        'docstrcbs': '\tCall-back functions:',
        'docreturn': '', 'docsign': '', 'docsignopt': '',
        'latexdocstrreq': '\\noindent Required arguments:',
        'latexdocstropt': '\\noindent Optional arguments:',
        'latexdocstrout': '\\noindent Return objects:',
        'latexdocstrcbs': '\\noindent Call-back functions:',
    }, {
        'frompyobj': [{debugcapi: '    CFUNCSMESS("cb:Getting return_value->");'},
                      {debugcapi:
                      ],
        'return': '    return return_value;',
        '_check': l_and(isfunction, l_not(isstringfunction), l_not(iscomplexfunction))
    },
    {
        'args_nm': 'return_value,&return_value_len',
        'frompyobj': [{debugcapi: '    CFUNCSMESS("cb:Getting return_value->\\"");'},
                      """    if (capi_j>capi_i)
        GETSTRFROMPYTUPLE(capi_return,capi_i++,return_value,return_value_len),
        'frompyobj': [{debugcapi: '    CFUNCSMESS("cb:Getting return_value->");'},
                      """\
    if (capi_j>capi_i),
        '_check': iscomplexfunction
    },
     '_check': isfunction},
    {'_check': issubroutine, 'return': 'return;'}
]
cb_arg_rules = [
    {
                                           l_and(hasnote, isintent_nothide): '--- See above.'}]},
        'depend': ''
    },
    {
        'args': {
        },
        'args_nm': {
        },
        'args_td': {
        },
        'strarglens_td': {isstring: ',int'},
    },
    {
        'error': {l_and(isintent_c, isintent_out,
                        throw_error('intent(c,out) is forbidden for callback scalar arguments')):
                  ''},
                      {isintent_out:
                      {l_and(debugcapi, l_and(l_not(iscomplex), isintent_c)):
                      {l_and(debugcapi, l_and(l_not(iscomplex), l_not( isintent_c))):
                      {l_and(debugcapi, l_and(iscomplex, isintent_c)):
                      {l_and(debugcapi, l_and(iscomplex, l_not( isintent_c))):
                      ],
                 {debugcapi: 'CFUNCSMESS'}],
        '_check': isscalar
    }, {
        'pyobjfrom': [{isintent_in: """\
    if (cb->nofargs>capi_i)
            goto capi_fail\
    if (cb->nofargs>capi_i)
            goto capi_fail;"""}],
        '_check': l_and(isscalar, isintent_nothide),
        '_optional': ''
    }, {
                      """    if (capi_j>capi_i)
                      {debugcapi:
                      ],
                 {debugcapi: 'CFUNCSMESS'}, 'string.h'],
        '_check': l_and(isstring, isintent_out)
    }, {
                      {isintent_in: """\
    if (cb->nofargs>capi_i)
            goto capi_fail\
    if (cb->nofargs>capi_i) {
            goto capi_fail;
    }"""}],
        '_check': l_and(isstring, isintent_nothide),
        '_optional': ''
    },
    {
        '_check': isarray,
        '_depend': ''
    },
    {
                      {isintent_c: """\
    if (cb->nofargs>capi_i) {
        /*XXX: Hmm, what will destroy this array??? */
""",
                       l_not(isintent_c): """\
    if (cb->nofargs>capi_i) {
        /*XXX: Hmm, what will destroy this array??? */
        if (tmp_arr==NULL)
            goto capi_fail;
        if (CAPI_ARGLIST_SETITEM(capi_i++,(PyObject *)tmp_arr))
            goto capi_fail;
}"""],
        '_check': l_and(isarray, isintent_nothide, l_or(isintent_in, isintent_inout)),
        '_optional': '',
    }, {
                      """    if (capi_j>capi_i) {
        PyArrayObject *rv_cb_arr = NULL;
        if ((capi_tmp = PyTuple_GetItem(capi_return,capi_i++))==NULL) goto capi_fail;
                      {isintent_c: '|F2PY_INTENT_C'},
                      """,capi_tmp);
        if (rv_cb_arr == NULL) {
            fprintf(stderr,\"rv_cb_arr is NULL\\n\");
            goto capi_fail;
        }
        if (capi_tmp != (PyObject *)rv_cb_arr) {
            Py_DECREF(rv_cb_arr);
        }
    }""",
                      {debugcapi: '    fprintf(stderr,"<-.\\n");'},
                      ],
        '_check': l_and(isarray, isintent_out)
    }, {
        '_check': isintent_out
    }
]
cb_map = {}
def buildcallbacks(m):
    cb_map[m['name']] = []
    for bi in m['body']:
        if bi['block'] == 'interface':
            for b in bi['body']:
                if b:
                    buildcallback(b, m['name'])
                else:
                    errmess('warning: empty body for %s\n' % (m['name']))
def buildcallback(rout, um):
    from . import capi_maps
    outmess('\tConstructing call-back function "cb_%s_in_%s"\n' %
            (rout['name'], um))
    args, depargs = getargs(rout)
    capi_maps.depargs = depargs
    var = rout['vars']
    vrd = capi_maps.cb_routsign2map(rout, um)
    rd = dictappend({}, vrd)
    cb_map[um].append([rout['name'], rd['name']])
    for r in cb_rout_rules:
        if ('_check' in r and r['_check'](rout)) or ('_check' not in r):
            ar = applyrules(r, vrd, rout)
            rd = dictappend(rd, ar)
    savevrd = {}
    for i, a in enumerate(args):
        vrd = capi_maps.cb_sign2map(a, var[a], index=i)
        savevrd[a] = vrd
        for r in cb_arg_rules:
            if '_depend' in r:
                continue
            if '_optional' in r and isoptional(var[a]):
                continue
            if ('_check' in r and r['_check'](var[a])) or ('_check' not in r):
                ar = applyrules(r, vrd, var[a])
                rd = dictappend(rd, ar)
                if '_break' in r:
                    break
    for a in args:
        vrd = savevrd[a]
        for r in cb_arg_rules:
            if '_depend' in r:
                continue
            if ('_optional' not in r) or ('_optional' in r and isrequired(var[a])):
                continue
            if ('_check' in r and r['_check'](var[a])) or ('_check' not in r):
                ar = applyrules(r, vrd, var[a])
                rd = dictappend(rd, ar)
                if '_break' in r:
                    break
    for a in depargs:
        vrd = savevrd[a]
        for r in cb_arg_rules:
            if '_depend' not in r:
                continue
            if '_optional' in r:
                continue
            if ('_check' in r and r['_check'](var[a])) or ('_check' not in r):
                ar = applyrules(r, vrd, var[a])
                rd = dictappend(rd, ar)
                if '_break' in r:
                    break
    if 'args' in rd and 'optargs' in rd:
        if isinstance(rd['optargs'], list):
            rd['optargs'] = rd['optargs'] + ]
    if isinstance(rd['docreturn'], list):
        rd['docreturn'] = stripcomma(
                                 {'docsignopt': rd['docsignopt']}
                                 ))
    if optargs == '':
        rd['docsignature'] = stripcomma(
    else:
                                     {'docsign': rd['docsign'],
                                      'docsignopt': optargs,
                                      })
    rd['latexdocsignature'] = rd['docsignature'].replace('_', '\\_')
    rd['latexdocsignature'] = rd['latexdocsignature'].replace(',', ', ')
    rd['docstrsigns'] = []
    rd['latexdocstrsigns'] = []
    for k in ['docstrreq', 'docstropt', 'docstrout', 'docstrcbs']:
        if k in rd and isinstance(rd[k], list):
            rd['docstrsigns'] = rd['docstrsigns'] + rd[k]
        k = 'latex' + k
        if k in rd and isinstance(rd[k], list):
            rd['latexdocstrsigns'] = rd['latexdocstrsigns'] + rd[k][0:1] +                ['\\begin{description}'] + rd[k][1:] +                ['\\end{description}']
    if 'args' not in rd:
        rd['args'] = ''
        rd['args_td'] = ''
        rd['args_nm'] = ''
    if not (rd.get('args') or rd.get('optargs') or rd.get('strarglens')):
        rd['noargs'] = 'void'
    ar = applyrules(cb_routine_rules, rd)
    cfuncs.callbacks[rd['name']] = ar['body']
    if isinstance(ar['need'], str):
        ar['need'] = [ar['need']]
    if 'need' in rd:
        for t in cfuncs.typedefs.keys():
            if t in rd['need']:
                ar['need'].append(t)
    cfuncs.typedefs_generated[rd['name'] + '_typedef'] = ar['cbtypedefs']
    ar['need'].append(rd['name'] + '_typedef')
    cfuncs.needs[rd['name']] = ar['need']
    capi_maps.lcb2_map[rd['name']] = {'maxnofargs': ar['maxnofargs'],
                                      'nofoptargs': ar['nofoptargs'],
                                      'docstr': ar['docstr'],
                                      'latexdocstr': ar['latexdocstr'],
                                      'argname': rd['argname']
                                      }
    outmess('\t  %s\n' % (ar['docstrshort']))
    return
