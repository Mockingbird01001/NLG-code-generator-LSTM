\
    PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\\
    fprintf(stderr,\"\\n\");\
 (size_t)(b) = ((size_t)(a) ^ (size_t)(b));\\
 (size_t)(a) = ((size_t)(a) ^ (size_t)(b))\
    PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\\
    fprintf(stderr,\"\\n\");\
static int f2py_size(PyArrayObject* var, ...)
{
  npy_int sz = 0;
  npy_int dim;
  npy_int rank;
  va_list argp;
  va_start(argp, var);
  dim = va_arg(argp, npy_int);
  if (dim==-1)
    {
      sz = PyArray_SIZE(var);
    }
  else
    {
      rank = PyArray_NDIM(var);
      if (dim>=1 && dim<=rank)
        sz = PyArray_DIM(var, dim-1);
      else
        fprintf(stderr, \"f2py_size: 2nd argument value=%d fails to satisfy 1<=value<=%d. Result will be 0.\\n\", dim, rank);
    }
  va_end(argp);
  return sz;
}\
/* New SciPy */
        PyArrayObject *arr = NULL;\\
        if (!obj) return -2;\\
        if (!PyArray_Check(obj)) return -1;\\
        if (!(arr=(PyArrayObject *)obj)) {fprintf(stderr,\"TRYPYARRAYTEMPLATE:\");PRINTPYOBJERR(obj);return 0;}\\
        if (PyArray_DESCR(arr)->type==typecode)  {*(ctype *)(PyArray_DATA(arr))=*v; return 1;}\\
        switch (PyArray_TYPE(arr)) {\\
                case NPY_DOUBLE: *(double *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_INT: *(int *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_LONG: *(long *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_FLOAT: *(float *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_CDOUBLE: *(double *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_CFLOAT: *(float *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_BOOL: *(npy_bool *)(PyArray_DATA(arr))=(*v!=0); break;\\
                case NPY_UBYTE: *(unsigned char *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_BYTE: *(signed char *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_SHORT: *(short *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_USHORT: *(npy_ushort *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_UINT: *(npy_uint *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_ULONG: *(npy_ulong *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_LONGLONG: *(npy_longlong *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_ULONGLONG: *(npy_ulonglong *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_LONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_CLONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=*v; break;\\
        default: return -2;\\
        };\\
        return 1\
        PyArrayObject *arr = NULL;\\
        if (!obj) return -2;\\
        if (!PyArray_Check(obj)) return -1;\\
        if (!(arr=(PyArrayObject *)obj)) {fprintf(stderr,\"TRYCOMPLEXPYARRAYTEMPLATE:\");PRINTPYOBJERR(obj);return 0;}\\
        if (PyArray_DESCR(arr)->type==typecode) {\\
            *(ctype *)(PyArray_DATA(arr))=(*v).r;\\
            *(ctype *)(PyArray_DATA(arr)+sizeof(ctype))=(*v).i;\\
            return 1;\\
        }\\
        switch (PyArray_TYPE(arr)) {\\
                case NPY_CDOUBLE: *(double *)(PyArray_DATA(arr))=(*v).r;*(double *)(PyArray_DATA(arr)+sizeof(double))=(*v).i;break;\\
                case NPY_CFLOAT: *(float *)(PyArray_DATA(arr))=(*v).r;*(float *)(PyArray_DATA(arr)+sizeof(float))=(*v).i;break;\\
                case NPY_DOUBLE: *(double *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_LONG: *(long *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_FLOAT: *(float *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_INT: *(int *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_SHORT: *(short *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_UBYTE: *(unsigned char *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_BYTE: *(signed char *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_BOOL: *(npy_bool *)(PyArray_DATA(arr))=((*v).r!=0 && (*v).i!=0); break;\\
                case NPY_USHORT: *(npy_ushort *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_UINT: *(npy_uint *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_ULONG: *(npy_ulong *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_LONGLONG: *(npy_longlong *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_ULONGLONG: *(npy_ulonglong *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_LONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_CLONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=(*v).r;*(npy_longdouble *)(PyArray_DATA(arr)+sizeof(npy_longdouble))=(*v).i;break;\\
                default: return -2;\\
        };\\
        return -1;\
        PyObject *rv_cb_str = PyTuple_GetItem((tuple),(index));\\
        if (rv_cb_str == NULL)\\
            goto capi_fail;\\
        if (PyBytes_Check(rv_cb_str)) {\\
            str[len-1]='\\0';\\
            STRINGCOPYN((str),PyBytes_AS_STRING((PyBytesObject*)rv_cb_str),(len));\\
        } else {\\
            PRINTPYOBJERR(rv_cb_str);\\
            goto capi_fail;\\
        }\\
    }\
        if ((capi_tmp = PyTuple_GetItem((tuple),(index)))==NULL) goto capi_fail;\\
            goto capi_fail;\\
    }\\
    if ((p) == NULL) {                                              \\
        PyErr_SetString(PyExc_MemoryError, "NULL pointer found");   \\
        goto capi_fail;                                             \\
    }                                                               \\
} while (0)\
    do { FAILNULL(to); FAILNULL(from); (void)memcpy(to,from,n); } while (0)\
    if ((str = (string)malloc(sizeof(char)*(len+1))) == NULL) {\\
        PyErr_SetString(PyExc_MemoryError, \"out of memory\");\\
        goto capi_fail;\\
    } else {\\
        (str)[len] = '\\0';\\
    }\
    do {                                                        \\
        int _m = (buf_size);                                    \\
        char *_to = (to);                                       \\
        char *_from = (from);                                   \\
        FAILNULL(_to); FAILNULL(_from);                         \\
        (void)strncpy(_to, _from, sizeof(char)*_m);             \\
        _to[_m-1] = '\\0';                                      \\
        /* Padding with spaces instead of nulls */              \\
        for (_m -= 2; _m >= 0 && _to[_m] == '\\0'; _m--) {      \\
            _to[_m] = ' ';                                      \\
        }                                                       \\
    } while (0)\
    do { FAILNULL(to); FAILNULL(from); (void)strcpy(to,from); } while (0)\
    if (!(check)) {\\
        /*goto capi_fail;*/\\
    } else\
    if (!(check)) {\\
        /*goto capi_fail;*/\\
    } else\
    if (!(check)) {\\
        char errstring[256];\\
        sprintf(errstring, \"%s: \"show, \"(\"tcheck\") failed for \"name, slen(var), var);\\
        /*goto capi_fail;*/\\
    } else\
    if (!(check)) {\\
        char errstring[256];\\
        sprintf(errstring, \"%s: \"show, \"(\"tcheck\") failed for \"name, var);\\
        /*goto capi_fail;*/\\
    } else\
      || defined(_WIN32) || defined(_WIN64) \\
      || defined(__MINGW32__) || defined(__MINGW64__)
      && (__STDC_VERSION__ >= 201112L) \\
      && !defined(__STDC_NO_THREADS__) \\
      && (!defined(__GLIBC__) || __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 12))
/* __STDC_NO_THREADS__ was first defined in a maintenance release of glibc 2.12,
   see https://lists.gnu.org/archive/html/commit-hurd/2012-07/msg00180.html,
   so `!defined(__STDC_NO_THREADS__)` may give false positive for the existence
   of `threads.h` when using an older release of glibc 2.12 */
      && (__GNUC__ > 4 || (__GNUC__ == 4 && (__GNUC_MINOR__ >= 4)))\
static int calcarrindex(int *i,PyArrayObject *arr) {
    int k,ii = i[0];
    for (k=1; k < PyArray_NDIM(arr); k++)
        ii += (ii*(PyArray_DIM(arr,k) - 1)+i[k]); /* assuming contiguous arr */
    return ii;
\
static int calcarrindextr(int *i,PyArrayObject *arr) {
    int k,ii = i[PyArray_NDIM(arr)-1];
    for (k=1; k < PyArray_NDIM(arr); k++)
        ii += (ii*(PyArray_DIM(arr,PyArray_NDIM(arr)-k-1) - 1)+i[PyArray_NDIM(arr)-k-1]); /* assuming contiguous arr */
    return ii;
\
static struct { int nd;npy_intp *d;int *i,*i_tr,tr; } forcombcache;
static int initforcomb(npy_intp *dims,int nd,int tr) {
  int k;
  if (dims==NULL) return 0;
  if (nd<0) return 0;
  forcombcache.nd = nd;
  forcombcache.d = dims;
  forcombcache.tr = tr;
  if ((forcombcache.i = (int *)malloc(sizeof(int)*nd))==NULL) return 0;
  if ((forcombcache.i_tr = (int *)malloc(sizeof(int)*nd))==NULL) return 0;
  for (k=1;k<nd;k++) {
    forcombcache.i[k] = forcombcache.i_tr[nd-k-1] = 0;
  }
  forcombcache.i[0] = forcombcache.i_tr[nd-1] = -1;
  return 1;
}
static int *nextforcomb(void) {
  int j,*i,*i_tr,k;
  int nd=forcombcache.nd;
  if ((i=forcombcache.i) == NULL) return NULL;
  if ((i_tr=forcombcache.i_tr) == NULL) return NULL;
  if (forcombcache.d == NULL) return NULL;
  i[0]++;
  if (i[0]==forcombcache.d[0]) {
    j=1;
    while ((j<nd) && (i[j]==forcombcache.d[j]-1)) j++;
    if (j==nd) {
      free(i);
      free(i_tr);
      return NULL;
    }
    for (k=0;k<j;k++) i[k] = i_tr[nd-k-1] = 0;
    i[j]++;
    i_tr[nd-j-1]++;
  } else
    i_tr[nd-1]++;
  if (forcombcache.tr) return i_tr;
  return i;
\
static int try_pyarr_from_string(PyObject *obj,const string str) {
    PyArrayObject *arr = NULL;
    if (PyArray_Check(obj) && (!((arr = (PyArrayObject *)obj) == NULL)))
        { STRINGCOPYN(PyArray_DATA(arr),str,PyArray_NBYTES(arr)); }
    return 1;
capi_fail:
    PRINTPYOBJERR(obj);
    return 0;
}\
static int
string_from_pyobj(string *str,int *len,const string inistr,PyObject *obj,const char *errmess)
{
    PyArrayObject *arr = NULL;
    PyObject *tmp = NULL;
fprintf(stderr,\"string_from_pyobj(str='%s',len=%d,inistr='%s',obj=%p)\\n\",(char*)str,*len,(char *)inistr,obj);
    if (obj == Py_None) {
        if (*len == -1)
            *len = strlen(inistr); /* Will this cause problems? */
        STRINGMALLOC(*str,*len);
        STRINGCOPYN(*str,inistr,*len+1);
        return 1;
    }
    if (PyArray_Check(obj)) {
        if ((arr = (PyArrayObject *)obj) == NULL)
            goto capi_fail;
        if (!ISCONTIGUOUS(arr)) {
            PyErr_SetString(PyExc_ValueError,\"array object is non-contiguous.\");
            goto capi_fail;
        }
        if (*len == -1)
            *len = (PyArray_ITEMSIZE(arr))*PyArray_SIZE(arr);
        STRINGMALLOC(*str,*len);
        STRINGCOPYN(*str,PyArray_DATA(arr),*len+1);
        return 1;
    }
    if (PyBytes_Check(obj)) {
        tmp = obj;
        Py_INCREF(tmp);
    }
    else if (PyUnicode_Check(obj)) {
        tmp = PyUnicode_AsASCIIString(obj);
    }
    else {
        PyObject *tmp2;
        tmp2 = PyObject_Str(obj);
        if (tmp2) {
            tmp = PyUnicode_AsASCIIString(tmp2);
            Py_DECREF(tmp2);
        }
        else {
            tmp = NULL;
        }
    }
    if (tmp == NULL) goto capi_fail;
    if (*len == -1)
        *len = PyBytes_GET_SIZE(tmp);
    STRINGMALLOC(*str,*len);
    STRINGCOPYN(*str,PyBytes_AS_STRING(tmp),*len+1);
    Py_DECREF(tmp);
    return 1;
capi_fail:
    Py_XDECREF(tmp);
    {
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
        }
        PyErr_SetString(err, errmess);
    }
    return 0;
}\
static int
char_from_pyobj(char* v, PyObject *obj, const char *errmess) {
    int i = 0;
    if (int_from_pyobj(&i, obj, errmess)) {
        *v = (char)i;
        return 1;
    }
    return 0;
}\
static int
signed_char_from_pyobj(signed_char* v, PyObject *obj, const char *errmess) {
    int i = 0;
    if (int_from_pyobj(&i, obj, errmess)) {
        *v = (signed_char)i;
        return 1;
    }
    return 0;
}\
static int
short_from_pyobj(short* v, PyObject *obj, const char *errmess) {
    int i = 0;
    if (int_from_pyobj(&i, obj, errmess)) {
        *v = (short)i;
        return 1;
    }
    return 0;
}\
static int
int_from_pyobj(int* v, PyObject *obj, const char *errmess)
{
    PyObject* tmp = NULL;
    if (PyLong_Check(obj)) {
        *v = Npy__PyLong_AsInt(obj);
        return !(*v == -1 && PyErr_Occurred());
    }
    tmp = PyNumber_Long(obj);
    if (tmp) {
        *v = Npy__PyLong_AsInt(tmp);
        Py_DECREF(tmp);
        return !(*v == -1 && PyErr_Occurred());
    }
    if (PyComplex_Check(obj))
        tmp = PyObject_GetAttrString(obj,\"real\");
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj))
        /*pass*/;
    else if (PySequence_Check(obj))
        tmp = PySequence_GetItem(obj, 0);
    if (tmp) {
        PyErr_Clear();
        if (int_from_pyobj(v, tmp, errmess)) {
            Py_DECREF(tmp);
            return 1;
        }
        Py_DECREF(tmp);
    }
    {
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
        }
        PyErr_SetString(err, errmess);
    }
    return 0;
}\
static int
long_from_pyobj(long* v, PyObject *obj, const char *errmess) {
    PyObject* tmp = NULL;
    if (PyLong_Check(obj)) {
        *v = PyLong_AsLong(obj);
        return !(*v == -1 && PyErr_Occurred());
    }
    tmp = PyNumber_Long(obj);
    if (tmp) {
        *v = PyLong_AsLong(tmp);
        Py_DECREF(tmp);
        return !(*v == -1 && PyErr_Occurred());
    }
    if (PyComplex_Check(obj))
        tmp = PyObject_GetAttrString(obj,\"real\");
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj))
        /*pass*/;
    else if (PySequence_Check(obj))
        tmp = PySequence_GetItem(obj,0);
    if (tmp) {
        PyErr_Clear();
        if (long_from_pyobj(v, tmp, errmess)) {
            Py_DECREF(tmp);
            return 1;
        }
        Py_DECREF(tmp);
    }
    {
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
        }
        PyErr_SetString(err, errmess);
    }
    return 0;
}\
static int
long_long_from_pyobj(long_long* v, PyObject *obj, const char *errmess)
{
    PyObject* tmp = NULL;
    if (PyLong_Check(obj)) {
        *v = PyLong_AsLongLong(obj);
        return !(*v == -1 && PyErr_Occurred());
    }
    tmp = PyNumber_Long(obj);
    if (tmp) {
        *v = PyLong_AsLongLong(tmp);
        Py_DECREF(tmp);
        return !(*v == -1 && PyErr_Occurred());
    }
    if (PyComplex_Check(obj))
        tmp = PyObject_GetAttrString(obj,\"real\");
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj))
        /*pass*/;
    else if (PySequence_Check(obj))
        tmp = PySequence_GetItem(obj,0);
    if (tmp) {
        PyErr_Clear();
        if (long_long_from_pyobj(v, tmp, errmess)) {
            Py_DECREF(tmp);
            return 1;
        }
        Py_DECREF(tmp);
    }
    {
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
        }
        PyErr_SetString(err,errmess);
    }
    return 0;
}\
static int
long_double_from_pyobj(long_double* v, PyObject *obj, const char *errmess)
{
    double d=0;
    if (PyArray_CheckScalar(obj)){
        if PyArray_IsScalar(obj, LongDouble) {
            PyArray_ScalarAsCtype(obj, v);
            return 1;
        }
        else if (PyArray_Check(obj) && PyArray_TYPE(obj) == NPY_LONGDOUBLE) {
            (*v) = *((npy_longdouble *)PyArray_DATA(obj));
            return 1;
        }
    }
    if (double_from_pyobj(&d, obj, errmess)) {
        *v = (long_double)d;
        return 1;
    }
    return 0;
}\
static int
double_from_pyobj(double* v, PyObject *obj, const char *errmess)
{
    PyObject* tmp = NULL;
    if (PyFloat_Check(obj)) {
        *v = PyFloat_AsDouble(obj);
        return !(*v == -1.0 && PyErr_Occurred());
    }
    tmp = PyNumber_Float(obj);
    if (tmp) {
        *v = PyFloat_AsDouble(tmp);
        Py_DECREF(tmp);
        return !(*v == -1.0 && PyErr_Occurred());
    }
    if (PyComplex_Check(obj))
        tmp = PyObject_GetAttrString(obj,\"real\");
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj))
        /*pass*/;
    else if (PySequence_Check(obj))
        tmp = PySequence_GetItem(obj,0);
    if (tmp) {
        PyErr_Clear();
        if (double_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
        Py_DECREF(tmp);
    }
    {
        PyObject* err = PyErr_Occurred();
        PyErr_SetString(err,errmess);
    }
    return 0;
}\
static int
float_from_pyobj(float* v, PyObject *obj, const char *errmess)
{
    double d=0.0;
    if (double_from_pyobj(&d,obj,errmess)) {
        *v = (float)d;
        return 1;
    }
    return 0;
}\
static int
complex_long_double_from_pyobj(complex_long_double* v, PyObject *obj, const char *errmess)
{
    complex_double cd = {0.0,0.0};
    if (PyArray_CheckScalar(obj)){
        if PyArray_IsScalar(obj, CLongDouble) {
            PyArray_ScalarAsCtype(obj, v);
            return 1;
        }
        else if (PyArray_Check(obj) && PyArray_TYPE(obj)==NPY_CLONGDOUBLE) {
            (*v).r = ((npy_clongdouble *)PyArray_DATA(obj))->real;
            (*v).i = ((npy_clongdouble *)PyArray_DATA(obj))->imag;
            return 1;
        }
    }
    if (complex_double_from_pyobj(&cd,obj,errmess)) {
        (*v).r = (long_double)cd.r;
        (*v).i = (long_double)cd.i;
        return 1;
    }
    return 0;
}\
static int
complex_double_from_pyobj(complex_double* v, PyObject *obj, const char *errmess) {
    Py_complex c;
    if (PyComplex_Check(obj)) {
        c = PyComplex_AsCComplex(obj);
        (*v).r = c.real;
        (*v).i = c.imag;
        return 1;
    }
    if (PyArray_IsScalar(obj, ComplexFloating)) {
        if (PyArray_IsScalar(obj, CFloat)) {
            npy_cfloat new;
            PyArray_ScalarAsCtype(obj, &new);
            (*v).r = (double)new.real;
            (*v).i = (double)new.imag;
        }
        else if (PyArray_IsScalar(obj, CLongDouble)) {
            npy_clongdouble new;
            PyArray_ScalarAsCtype(obj, &new);
            (*v).r = (double)new.real;
            (*v).i = (double)new.imag;
        }
        else { /* if (PyArray_IsScalar(obj, CDouble)) */
            PyArray_ScalarAsCtype(obj, v);
        }
        return 1;
    }
    if (PyArray_CheckScalar(obj)) { /* 0-dim array or still array scalar */
        PyObject *arr;
        if (PyArray_Check(obj)) {
            arr = PyArray_Cast((PyArrayObject *)obj, NPY_CDOUBLE);
        }
        else {
            arr = PyArray_FromScalar(obj, PyArray_DescrFromType(NPY_CDOUBLE));
        }
        if (arr == NULL) {
            return 0;
        }
        (*v).r = ((npy_cdouble *)PyArray_DATA(arr))->real;
        (*v).i = ((npy_cdouble *)PyArray_DATA(arr))->imag;
        Py_DECREF(arr);
        return 1;
    }
    /* Python does not provide PyNumber_Complex function :-( */
    (*v).i = 0.0;
    if (PyFloat_Check(obj)) {
        (*v).r = PyFloat_AsDouble(obj);
        return !((*v).r == -1.0 && PyErr_Occurred());
    }
    if (PyLong_Check(obj)) {
        (*v).r = PyLong_AsDouble(obj);
        return !((*v).r == -1.0 && PyErr_Occurred());
    }
    if (PySequence_Check(obj) && !(PyBytes_Check(obj) || PyUnicode_Check(obj))) {
        PyObject *tmp = PySequence_GetItem(obj,0);
        if (tmp) {
            if (complex_double_from_pyobj(v,tmp,errmess)) {
                Py_DECREF(tmp);
                return 1;
            }
            Py_DECREF(tmp);
        }
    }
    {
        PyObject* err = PyErr_Occurred();
        if (err==NULL)
            err = PyExc_TypeError;
        PyErr_SetString(err,errmess);
    }
    return 0;
}\
static int
complex_float_from_pyobj(complex_float* v,PyObject *obj,const char *errmess)
{
    complex_double cd={0.0,0.0};
    if (complex_double_from_pyobj(&cd,obj,errmess)) {
        (*v).r = (float)cd.r;
        (*v).i = (float)cd.i;
        return 1;
    }
    return 0;
}
"""
needs['try_pyarr_from_char'] = ['pyobj_from_char1', 'TRYPYARRAYTEMPLATE']
cfuncs[
    'try_pyarr_from_char'] = 'static int try_pyarr_from_char(PyObject* obj,char* v) {\n    TRYPYARRAYTEMPLATE(char,\'c\');\n}\n'
needs['try_pyarr_from_signed_char'] = ['TRYPYARRAYTEMPLATE', 'unsigned_char']
cfuncs[
    'try_pyarr_from_unsigned_char'] = 'static int try_pyarr_from_unsigned_char(PyObject* obj,unsigned_char* v) {\n    TRYPYARRAYTEMPLATE(unsigned_char,\'b\');\n}\n'
needs['try_pyarr_from_signed_char'] = ['TRYPYARRAYTEMPLATE', 'signed_char']
cfuncs[
    'try_pyarr_from_signed_char'] = 'static int try_pyarr_from_signed_char(PyObject* obj,signed_char* v) {\n    TRYPYARRAYTEMPLATE(signed_char,\'1\');\n}\n'
needs['try_pyarr_from_short'] = ['pyobj_from_short1', 'TRYPYARRAYTEMPLATE']
cfuncs[
    'try_pyarr_from_short'] = 'static int try_pyarr_from_short(PyObject* obj,short* v) {\n    TRYPYARRAYTEMPLATE(short,\'s\');\n}\n'
needs['try_pyarr_from_int'] = ['pyobj_from_int1', 'TRYPYARRAYTEMPLATE']
cfuncs[
    'try_pyarr_from_int'] = 'static int try_pyarr_from_int(PyObject* obj,int* v) {\n    TRYPYARRAYTEMPLATE(int,\'i\');\n}\n'
needs['try_pyarr_from_long'] = ['pyobj_from_long1', 'TRYPYARRAYTEMPLATE']
cfuncs[
    'try_pyarr_from_long'] = 'static int try_pyarr_from_long(PyObject* obj,long* v) {\n    TRYPYARRAYTEMPLATE(long,\'l\');\n}\n'
needs['try_pyarr_from_long_long'] = [
    'pyobj_from_long_long1', 'TRYPYARRAYTEMPLATE', 'long_long']
cfuncs[
    'try_pyarr_from_long_long'] = 'static int try_pyarr_from_long_long(PyObject* obj,long_long* v) {\n    TRYPYARRAYTEMPLATE(long_long,\'L\');\n}\n'
needs['try_pyarr_from_float'] = ['pyobj_from_float1', 'TRYPYARRAYTEMPLATE']
cfuncs[
    'try_pyarr_from_float'] = 'static int try_pyarr_from_float(PyObject* obj,float* v) {\n    TRYPYARRAYTEMPLATE(float,\'f\');\n}\n'
needs['try_pyarr_from_double'] = ['pyobj_from_double1', 'TRYPYARRAYTEMPLATE']
cfuncs[
    'try_pyarr_from_double'] = 'static int try_pyarr_from_double(PyObject* obj,double* v) {\n    TRYPYARRAYTEMPLATE(double,\'d\');\n}\n'
needs['try_pyarr_from_complex_float'] = [
    'pyobj_from_complex_float1', 'TRYCOMPLEXPYARRAYTEMPLATE', 'complex_float']
cfuncs[
    'try_pyarr_from_complex_float'] = 'static int try_pyarr_from_complex_float(PyObject* obj,complex_float* v) {\n    TRYCOMPLEXPYARRAYTEMPLATE(float,\'F\');\n}\n'
needs['try_pyarr_from_complex_double'] = [
    'pyobj_from_complex_double1', 'TRYCOMPLEXPYARRAYTEMPLATE', 'complex_double']
cfuncs[
    'try_pyarr_from_complex_double'] = 'static int try_pyarr_from_complex_double(PyObject* obj,complex_double* v) {\n    TRYCOMPLEXPYARRAYTEMPLATE(double,\'D\');\n}\n'
needs['create_cb_arglist'] = ['CFUNCSMESS', 'PRINTPYOBJERR', 'MINMAX']
cfuncs['create_cb_arglist'] = """\
static int
create_cb_arglist(PyObject* fun, PyTupleObject* xa , const int maxnofargs,
                  const int nofoptargs, int *nofargs, PyTupleObject **args,
                  const char *errmess)
{
    PyObject *tmp = NULL;
    PyObject *tmp_fun = NULL;
    Py_ssize_t tot, opt, ext, siz, i, di = 0;
    CFUNCSMESS(\"create_cb_arglist\\n\");
    tot=opt=ext=siz=0;
    /* Get the total number of arguments */
    if (PyFunction_Check(fun)) {
        tmp_fun = fun;
        Py_INCREF(tmp_fun);
    }
    else {
        di = 1;
        if (PyObject_HasAttrString(fun,\"im_func\")) {
            tmp_fun = PyObject_GetAttrString(fun,\"im_func\");
        }
        else if (PyObject_HasAttrString(fun,\"__call__\")) {
            tmp = PyObject_GetAttrString(fun,\"__call__\");
            if (PyObject_HasAttrString(tmp,\"im_func\"))
                tmp_fun = PyObject_GetAttrString(tmp,\"im_func\");
            else {
                tmp_fun = fun; /* built-in function */
                Py_INCREF(tmp_fun);
                tot = maxnofargs;
                if (PyCFunction_Check(fun)) {
                    /* In case the function has a co_argcount (like on PyPy) */
                    di = 0;
                }
                if (xa != NULL)
                    tot += PyTuple_Size((PyObject *)xa);
            }
            Py_XDECREF(tmp);
        }
        else if (PyFortran_Check(fun) || PyFortran_Check1(fun)) {
            tot = maxnofargs;
            if (xa != NULL)
                tot += PyTuple_Size((PyObject *)xa);
            tmp_fun = fun;
            Py_INCREF(tmp_fun);
        }
        else if (F2PyCapsule_Check(fun)) {
            tot = maxnofargs;
            if (xa != NULL)
                ext = PyTuple_Size((PyObject *)xa);
            if(ext>0) {
                fprintf(stderr,\"extra arguments tuple cannot be used with CObject call-back\\n\");
                goto capi_fail;
            }
            tmp_fun = fun;
            Py_INCREF(tmp_fun);
        }
    }
    if (tmp_fun == NULL) {
        fprintf(stderr,
                \"Call-back argument must be function|instance|instance.__call__|f2py-function \"
                \"but got %s.\\n\",
                ((fun == NULL) ? \"NULL\" : Py_TYPE(fun)->tp_name));
        goto capi_fail;
    }
    if (PyObject_HasAttrString(tmp_fun,\"__code__\")) {
        if (PyObject_HasAttrString(tmp = PyObject_GetAttrString(tmp_fun,\"__code__\"),\"co_argcount\")) {
            PyObject *tmp_argcount = PyObject_GetAttrString(tmp,\"co_argcount\");
            Py_DECREF(tmp);
            if (tmp_argcount == NULL) {
                goto capi_fail;
            }
            tot = PyLong_AsSsize_t(tmp_argcount) - di;
            Py_DECREF(tmp_argcount);
        }
    }
    /* Get the number of optional arguments */
    if (PyObject_HasAttrString(tmp_fun,\"__defaults__\")) {
        if (PyTuple_Check(tmp = PyObject_GetAttrString(tmp_fun,\"__defaults__\")))
            opt = PyTuple_Size(tmp);
        Py_XDECREF(tmp);
    }
    /* Get the number of extra arguments */
    if (xa != NULL)
        ext = PyTuple_Size((PyObject *)xa);
    /* Calculate the size of call-backs argument list */
    siz = MIN(maxnofargs+ext,tot);
    *nofargs = MAX(0,siz-ext);
    fprintf(stderr,
            \"debug-capi:create_cb_arglist:maxnofargs(-nofoptargs),\"
            \"tot,opt,ext,siz,nofargs = %d(-%d), %zd, %zd, %zd, %zd, %d\\n\",
            maxnofargs, nofoptargs, tot, opt, ext, siz, *nofargs);
    if (siz < tot-opt) {
        fprintf(stderr,
                \"create_cb_arglist: Failed to build argument list \"
                \"(siz) with enough arguments (tot-opt) required by \"
                \"user-supplied function (siz,tot,opt=%zd, %zd, %zd).\\n\",
                siz, tot, opt);
        goto capi_fail;
    }
    /* Initialize argument list */
    *args = (PyTupleObject *)PyTuple_New(siz);
    for (i=0;i<*nofargs;i++) {
        Py_INCREF(Py_None);
        PyTuple_SET_ITEM((PyObject *)(*args),i,Py_None);
    }
    if (xa != NULL)
        for (i=(*nofargs);i<siz;i++) {
            tmp = PyTuple_GetItem((PyObject *)xa,i-(*nofargs));
            Py_INCREF(tmp);
            PyTuple_SET_ITEM(*args,i,tmp);
        }
    CFUNCSMESS(\"create_cb_arglist-end\\n\");
    Py_DECREF(tmp_fun);
    return 1;
capi_fail:
    if (PyErr_Occurred() == NULL)
    Py_XDECREF(tmp_fun);
    return 0;
}
"""
def buildcfuncs():
    from .capi_maps import c2capi_map
    for k in c2capi_map.keys():
        m = 'pyarr_from_p_%s1' % k
        cppmacros[
    k = 'string'
    m = 'pyarr_from_p_%s1' % k
    cppmacros[
def append_needs(need, flag=1):
    if isinstance(need, list):
        for n in need:
            append_needs(n, flag)
    elif isinstance(need, str):
        if not need:
            return
        if need in includes0:
            n = 'includes0'
        elif need in includes:
            n = 'includes'
        elif need in typedefs:
            n = 'typedefs'
        elif need in typedefs_generated:
            n = 'typedefs_generated'
        elif need in cppmacros:
            n = 'cppmacros'
        elif need in cfuncs:
            n = 'cfuncs'
        elif need in callbacks:
            n = 'callbacks'
        elif need in f90modhooks:
            n = 'f90modhooks'
        elif need in commonhooks:
            n = 'commonhooks'
        else:
            errmess('append_needs: unknown need %s\n' % (repr(need)))
            return
        if need in outneeds[n]:
            return
        if flag:
            tmp = {}
            if need in needs:
                for nn in needs[need]:
                    t = append_needs(nn, 0)
                    if isinstance(t, dict):
                        for nnn in t.keys():
                            if nnn in tmp:
                                tmp[nnn] = tmp[nnn] + t[nnn]
                            else:
                                tmp[nnn] = t[nnn]
            for nn in tmp.keys():
                for nnn in tmp[nn]:
                    if nnn not in outneeds[nn]:
                        outneeds[nn] = [nnn] + outneeds[nn]
            outneeds[n].append(need)
        else:
            tmp = {}
            if need in needs:
                for nn in needs[need]:
                    t = append_needs(nn, flag)
                    if isinstance(t, dict):
                        for nnn in t.keys():
                            if nnn in tmp:
                                tmp[nnn] = t[nnn] + tmp[nnn]
                            else:
                                tmp[nnn] = t[nnn]
            if n not in tmp:
                tmp[n] = []
            tmp[n].append(need)
            return tmp
    else:
        errmess('append_needs: expected list or string but got :%s\n' %
                (repr(need)))
def get_needs():
    res = {}
    for n in outneeds.keys():
        out = []
        saveout = copy.copy(outneeds[n])
        while len(outneeds[n]) > 0:
            if outneeds[n][0] not in needs:
                out.append(outneeds[n][0])
                del outneeds[n][0]
            else:
                flag = 0
                for k in outneeds[n][1:]:
                    if k in needs[outneeds[n][0]]:
                        flag = 1
                        break
                if flag:
                    outneeds[n] = outneeds[n][1:] + [outneeds[n][0]]
                else:
                    out.append(outneeds[n][0])
                    del outneeds[n][0]
            if saveout and (0 not in map(lambda x, y: x == y, saveout, outneeds[n]))                    and outneeds[n] != []:
                print(n, saveout)
                errmess(
                    'get_needs: no progress in sorting needs, probably circular dependence, skipping.\n')
                out = out + saveout
                break
            saveout = copy.copy(outneeds[n])
        if out == []:
            out = [n]
        res[n] = out
    return res
