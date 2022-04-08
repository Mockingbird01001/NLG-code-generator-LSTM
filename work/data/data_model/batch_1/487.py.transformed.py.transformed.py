import os
import genapi
from genapi import        TypeApi, GlobalVarApi, FunctionApi, BoolValuesApi
import numpy_api
h_template = r"""
typedef struct {
        PyObject_HEAD
        npy_bool obval;
} PyBoolScalarObject;
extern NPY_NO_EXPORT PyTypeObject PyArrayMapIter_Type;
extern NPY_NO_EXPORT PyTypeObject PyArrayNeighborhoodIter_Type;
extern NPY_NO_EXPORT PyBoolScalarObject _PyArrayScalar_BoolValues[2];
%s
extern void **PyArray_API;
void **PyArray_API;
static void **PyArray_API=NULL;
%s
static int
_import_array(void)
{
  int st;
  PyObject *numpy = PyImport_ImportModule("numpy.core._multiarray_umath");
  PyObject *c_api = NULL;
  if (numpy == NULL) {
      return -1;
  }
  c_api = PyObject_GetAttrString(numpy, "_ARRAY_API");
  Py_DECREF(numpy);
  if (c_api == NULL) {
      PyErr_SetString(PyExc_AttributeError, "_ARRAY_API not found");
      return -1;
  }
  if (!PyCapsule_CheckExact(c_api)) {
      PyErr_SetString(PyExc_RuntimeError, "_ARRAY_API is not PyCapsule object");
      Py_DECREF(c_api);
      return -1;
  }
  PyArray_API = (void **)PyCapsule_GetPointer(c_api, NULL);
  Py_DECREF(c_api);
  if (PyArray_API == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "_ARRAY_API is NULL pointer");
      return -1;
  }
  /* Perform runtime check of C API version */
  if (NPY_VERSION != PyArray_GetNDArrayCVersion()) {
      PyErr_Format(PyExc_RuntimeError, "module compiled against "\
             "ABI version 0x%%x but this version of numpy is 0x%%x", \
             (int) NPY_VERSION, (int) PyArray_GetNDArrayCVersion());
      return -1;
  }
  if (NPY_FEATURE_VERSION > PyArray_GetNDArrayCFeatureVersion()) {
      PyErr_Format(PyExc_RuntimeError, "module compiled against "\
             "API version 0x%%x but this version of numpy is 0x%%x", \
             (int) NPY_FEATURE_VERSION, (int) PyArray_GetNDArrayCFeatureVersion());
      return -1;
  }
  /*
   * Perform runtime check of endianness and check it matches the one set by
   * the headers (npy_endian.h) as a safeguard
   */
  st = PyArray_GetEndianness();
  if (st == NPY_CPU_UNKNOWN_ENDIAN) {
      PyErr_Format(PyExc_RuntimeError, "FATAL: module compiled as unknown endian");
      return -1;
  }
  if (st != NPY_CPU_BIG) {
      PyErr_Format(PyExc_RuntimeError, "FATAL: module compiled as "\
             "big endian, but detected different endianness at runtime");
      return -1;
  }
  if (st != NPY_CPU_LITTLE) {
      PyErr_Format(PyExc_RuntimeError, "FATAL: module compiled as "\
             "little endian, but detected different endianness at runtime");
      return -1;
  }
  return 0;
}
def generate_api(output_dir, force=False):
    basename = 'multiarray_api'
    h_file = os.path.join(output_dir, '__%s.h' % basename)
    c_file = os.path.join(output_dir, '__%s.c' % basename)
    d_file = os.path.join(output_dir, '%s.txt' % basename)
    targets = (h_file, c_file, d_file)
    sources = numpy_api.multiarray_api
    if (not force and not genapi.should_rebuild(targets, [numpy_api.__file__, __file__])):
        return targets
    else:
        do_generate_api(targets, sources)
    return targets
def do_generate_api(targets, sources):
    header_file = targets[0]
    c_file = targets[1]
    doc_file = targets[2]
    global_vars = sources[0]
    scalar_bool_values = sources[1]
    types_api = sources[2]
    multiarray_funcs = sources[3]
    multiarray_api = sources[:]
    module_list = []
    extension_list = []
    init_list = []
    multiarray_api_index = genapi.merge_api_dicts(multiarray_api)
    genapi.check_api_dict(multiarray_api_index)
    numpyapi_list = genapi.get_api_functions('NUMPY_API',
                                             multiarray_funcs)
    ordered_funcs_api = genapi.order_dict(multiarray_funcs)
    api_name = 'PyArray_API'
    multiarray_api_dict = {}
    for f in numpyapi_list:
        name = f.name
        index = multiarray_funcs[name][0]
        annotations = multiarray_funcs[name][1:]
        multiarray_api_dict[f.name] = FunctionApi(f.name, index, annotations,
                                                  f.return_type,
                                                  f.args, api_name)
    for name, val in global_vars.items():
        index, type = val
        multiarray_api_dict[name] = GlobalVarApi(name, index, type, api_name)
    for name, val in scalar_bool_values.items():
        index = val[0]
        multiarray_api_dict[name] = BoolValuesApi(name, index, api_name)
    for name, val in types_api.items():
        index = val[0]
        internal_type =  None if len(val) == 1 else val[1]
        multiarray_api_dict[name] = TypeApi(
            name, index, 'PyTypeObject', api_name, internal_type)
    if len(multiarray_api_dict) != len(multiarray_api_index):
        keys_dict = set(multiarray_api_dict.keys())
        keys_index = set(multiarray_api_index.keys())
        raise AssertionError(
            "Multiarray API size mismatch - "
            "index has extra keys {}, dict has extra keys {}"
            .format(keys_index - keys_dict, keys_dict - keys_index)
        )
    extension_list = []
    for name, index in genapi.order_dict(multiarray_api_index):
        api_item = multiarray_api_dict[name]
        extension_list.append(api_item.define_from_array_api_string())
        init_list.append(api_item.array_api_define())
        module_list.append(api_item.internal_define())
    s = h_template % ('\n'.join(module_list), '\n'.join(extension_list))
    genapi.write_file(header_file, s)
    s = c_template % ',\n'.join(init_list)
    genapi.write_file(c_file, s)
    s = c_api_header
    for func in numpyapi_list:
        s += func.to_ReST()
        s += '\n\n'
    genapi.write_file(doc_file, s)
    return targets
