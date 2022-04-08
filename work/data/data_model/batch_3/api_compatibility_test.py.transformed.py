
_API_GOLDEN_FOLDER_V1 = None
_API_GOLDEN_FOLDER_V2 = None
def _InitPathConstants():
  global _API_GOLDEN_FOLDER_V1
  global _API_GOLDEN_FOLDER_V2
  root_golden_path_v2 = os.path.join(resource_loader.get_data_files_path(),
                                     '..', 'golden', 'v2', 'tensorflow.pbtxt')
  if FLAGS.update_goldens:
    root_golden_path_v2 = os.path.realpath(root_golden_path_v2)
  _API_GOLDEN_FOLDER_V2 = os.path.dirname(root_golden_path_v2)
  _API_GOLDEN_FOLDER_V1 = os.path.normpath(
      os.path.join(_API_GOLDEN_FOLDER_V2, '..', 'v1'))
_TEST_README_FILE = resource_loader.get_path_to_datafile('README.txt')
_UPDATE_WARNING_FILE = resource_loader.get_path_to_datafile(
    'API_UPDATE_WARNING.txt')
_NON_CORE_PACKAGES = ['estimator', 'keras']
_V1_APIS_FROM_KERAS = ['layers', 'nn.rnn_cell']
_V2_APIS_FROM_KERAS = ['initializers', 'losses', 'metrics', 'optimizers']
if not hasattr(tf.compat.v1, 'estimator'):
  tf.compat.v1.estimator = tf.estimator
  tf.compat.v2.estimator = tf.estimator
def _KeyToFilePath(key, api_version):
  def _ReplaceCapsWithDash(matchobj):
    match = matchobj.group(0)
    return '-%s' % (match.lower())
  case_insensitive_key = re.sub('([A-Z]{1})', _ReplaceCapsWithDash,
                                six.ensure_str(key))
  api_folder = (
      _API_GOLDEN_FOLDER_V2 if api_version == 2 else _API_GOLDEN_FOLDER_V1)
  if key.startswith('tensorflow.experimental.numpy'):
    api_folder = os.path.join(
        api_folder, '..', '..', '..', '..', '../third_party',
        'py', 'numpy', 'tf_numpy_api')
    api_folder = os.path.normpath(api_folder)
  return os.path.join(api_folder, '%s.pbtxt' % case_insensitive_key)
def _FileNameToKey(filename):
  def _ReplaceDashWithCaps(matchobj):
    match = matchobj.group(0)
    return match[1].upper()
  base_filename = os.path.basename(filename)
  base_filename_without_ext = os.path.splitext(base_filename)[0]
  api_object_key = re.sub('((-[a-z]){1})', _ReplaceDashWithCaps,
                          six.ensure_str(base_filename_without_ext))
  return api_object_key
def _VerifyNoSubclassOfMessageVisitor(path, parent, unused_children):
  if not (isinstance(parent, type) and issubclass(parent, message.Message)):
    return
  if parent is message.Message:
    return
  if message.Message not in parent.__bases__:
    raise NotImplementedError(
        'Object tf.%s is a subclass of a generated proto Message. '
        'They are not yet supported by the API tools.' % path)
def _FilterNonCoreGoldenFiles(golden_file_list):
  return _FilterGoldenFilesByPrefix(golden_file_list, _NON_CORE_PACKAGES)
def _FilterV1KerasRelatedGoldenFiles(golden_file_list):
  return _FilterGoldenFilesByPrefix(golden_file_list, _V1_APIS_FROM_KERAS)
def _FilterV2KerasRelatedGoldenFiles(golden_file_list):
  return _FilterGoldenFilesByPrefix(golden_file_list, _V2_APIS_FROM_KERAS)
def _FilterGoldenFilesByPrefix(golden_file_list, package_prefixes):
  filtered_file_list = []
  filtered_package_prefixes = ['tensorflow.%s.' % p for p in package_prefixes]
  for f in golden_file_list:
    if any(
        six.ensure_str(f).rsplit('/')[-1].startswith(pre)
        for pre in filtered_package_prefixes):
      continue
    filtered_file_list.append(f)
  return filtered_file_list
def _FilterGoldenProtoDict(golden_proto_dict, omit_golden_symbols_map):
  if not omit_golden_symbols_map:
    return golden_proto_dict
  filtered_proto_dict = dict(golden_proto_dict)
  for key, symbol_list in six.iteritems(omit_golden_symbols_map):
    api_object = api_objects_pb2.TFAPIObject()
    api_object.CopyFrom(filtered_proto_dict[key])
    filtered_proto_dict[key] = api_object
    module_or_class = None
    if api_object.HasField('tf_module'):
      module_or_class = api_object.tf_module
    elif api_object.HasField('tf_class'):
      module_or_class = api_object.tf_class
    if module_or_class is not None:
      for members in (module_or_class.member, module_or_class.member_method):
        filtered_members = [m for m in members if m.name not in symbol_list]
        del members[:]
        members.extend(filtered_members)
  return filtered_proto_dict
def _GetTFNumpyGoldenPattern(api_version):
  return os.path.join(resource_loader.get_root_dir_with_all_resources(),
                      _KeyToFilePath('tensorflow.experimental.numpy*',
                                     api_version))
class ApiCompatibilityTest(test.TestCase):
  def __init__(self, *args, **kwargs):
    super(ApiCompatibilityTest, self).__init__(*args, **kwargs)
    golden_update_warning_filename = os.path.join(
        resource_loader.get_root_dir_with_all_resources(), _UPDATE_WARNING_FILE)
    self._update_golden_warning = file_io.read_file_to_string(
        golden_update_warning_filename)
    test_readme_filename = os.path.join(
        resource_loader.get_root_dir_with_all_resources(), _TEST_README_FILE)
    self._test_readme_message = file_io.read_file_to_string(
        test_readme_filename)
  def _AssertProtoDictEquals(self,
                             expected_dict,
                             actual_dict,
                             verbose=False,
                             update_goldens=False,
                             additional_missing_object_message='',
                             api_version=2):
    diffs = []
    verbose_diffs = []
    expected_keys = set(expected_dict.keys())
    actual_keys = set(actual_dict.keys())
    only_in_expected = expected_keys - actual_keys
    only_in_actual = actual_keys - expected_keys
    all_keys = expected_keys | actual_keys
    updated_keys = []
    for key in all_keys:
      diff_message = ''
      verbose_diff_message = ''
      if key in only_in_expected:
        diff_message = 'Object %s expected but not found (removed). %s' % (
            key, additional_missing_object_message)
        verbose_diff_message = diff_message
      elif key in only_in_actual:
        diff_message = 'New object %s found (added).' % key
        verbose_diff_message = diff_message
      else:
        try:
          self.assertProtoEquals(expected_dict[key], actual_dict[key])
        except AssertionError as e:
          updated_keys.append(key)
          diff_message = 'Change detected in python object: %s.' % key
          verbose_diff_message = str(e)
      if diff_message:
        diffs.append(diff_message)
        verbose_diffs.append(verbose_diff_message)
    if diffs:
      diff_count = len(diffs)
      logging.error(self._test_readme_message)
      logging.error('%d differences found between API and golden.', diff_count)
      if update_goldens:
        logging.warning(self._update_golden_warning)
        for key in only_in_expected:
          filepath = _KeyToFilePath(key, api_version)
          file_io.delete_file(filepath)
        for key in only_in_actual | set(updated_keys):
          filepath = _KeyToFilePath(key, api_version)
          file_io.write_string_to_file(
              filepath, text_format.MessageToString(actual_dict[key]))
      else:
        for d, verbose_d in zip(diffs, verbose_diffs):
          logging.error('    %s', d)
          logging.error('    %s', verbose_d)
        self.fail('%d differences found between API and golden.' % diff_count)
    else:
      logging.info('No differences found between API and golden.')
  def testNoSubclassOfMessage(self):
    visitor = public_api.PublicAPIVisitor(_VerifyNoSubclassOfMessageVisitor)
    visitor.do_not_descend_map['tf'].append('contrib')
    visitor.private_map['tf.compat'] = ['v1', 'v2']
    traverse.traverse(tf, visitor)
  def testNoSubclassOfMessageV1(self):
    if not hasattr(tf.compat, 'v1'):
      return
    visitor = public_api.PublicAPIVisitor(_VerifyNoSubclassOfMessageVisitor)
    visitor.do_not_descend_map['tf'].append('contrib')
    if FLAGS.only_test_core_api:
      visitor.do_not_descend_map['tf'].extend(_NON_CORE_PACKAGES)
    visitor.private_map['tf.compat'] = ['v1', 'v2']
    traverse.traverse(tf.compat.v1, visitor)
  def testNoSubclassOfMessageV2(self):
    if not hasattr(tf.compat, 'v2'):
      return
    visitor = public_api.PublicAPIVisitor(_VerifyNoSubclassOfMessageVisitor)
    visitor.do_not_descend_map['tf'].append('contrib')
    if FLAGS.only_test_core_api:
      visitor.do_not_descend_map['tf'].extend(_NON_CORE_PACKAGES)
    visitor.private_map['tf.compat'] = ['v1', 'v2']
    traverse.traverse(tf.compat.v2, visitor)
  def _checkBackwardsCompatibility(self,
                                   root,
                                   golden_file_patterns,
                                   api_version,
                                   additional_private_map=None,
                                   omit_golden_symbols_map=None):
    visitor = python_object_to_proto_visitor.PythonObjectToProtoVisitor()
    public_api_visitor = public_api.PublicAPIVisitor(visitor)
    public_api_visitor.private_map['tf'].append('contrib')
    if api_version == 2:
      public_api_visitor.private_map['tf'].append('enable_v2_behavior')
    public_api_visitor.do_not_descend_map['tf.GPUOptions'] = ['Experimental']
    public_api_visitor.do_not_descend_map['tf.experimental.numpy'] = [
        'bool_', 'complex_', 'complex128', 'complex64', 'float_', 'float16',
        'float32', 'float64', 'inexact', 'int_', 'int16', 'int32', 'int64',
        'int8', 'object_', 'string_', 'uint16', 'uint32', 'uint64', 'uint8',
        'unicode_', 'iinfo']
    public_api_visitor.do_not_descend_map['tf'].append('keras')
    if FLAGS.only_test_core_api:
      public_api_visitor.do_not_descend_map['tf'].extend(_NON_CORE_PACKAGES)
      if api_version == 2:
        public_api_visitor.do_not_descend_map['tf'].extend(_V2_APIS_FROM_KERAS)
      else:
        public_api_visitor.do_not_descend_map['tf'].extend(['layers'])
        public_api_visitor.do_not_descend_map['tf.nn'] = ['rnn_cell']
    if additional_private_map:
      public_api_visitor.private_map.update(additional_private_map)
    traverse.traverse(root, public_api_visitor)
    proto_dict = visitor.GetProtos()
    golden_file_list = file_io.get_matching_files(golden_file_patterns)
    if FLAGS.only_test_core_api:
      golden_file_list = _FilterNonCoreGoldenFiles(golden_file_list)
      if api_version == 2:
        golden_file_list = _FilterV2KerasRelatedGoldenFiles(golden_file_list)
      else:
        golden_file_list = _FilterV1KerasRelatedGoldenFiles(golden_file_list)
    def _ReadFileToProto(filename):
      ret_val = api_objects_pb2.TFAPIObject()
      text_format.Merge(file_io.read_file_to_string(filename), ret_val)
      return ret_val
    golden_proto_dict = {
        _FileNameToKey(filename): _ReadFileToProto(filename)
        for filename in golden_file_list
    }
    golden_proto_dict = _FilterGoldenProtoDict(golden_proto_dict,
                                               omit_golden_symbols_map)
    self._AssertProtoDictEquals(
        golden_proto_dict,
        proto_dict,
        verbose=FLAGS.verbose_diffs,
        update_goldens=FLAGS.update_goldens,
        api_version=api_version)
  def testAPIBackwardsCompatibility(self):
    api_version = 1
    if hasattr(tf, '_major_api_version') and tf._major_api_version == 2:
      api_version = 2
    golden_file_patterns = [
        os.path.join(resource_loader.get_root_dir_with_all_resources(),
                     _KeyToFilePath('*', api_version)),
        _GetTFNumpyGoldenPattern(api_version)]
    omit_golden_symbols_map = {}
    if (api_version == 2 and FLAGS.only_test_core_api and
        not _TENSORBOARD_AVAILABLE):
      omit_golden_symbols_map['tensorflow.summary'] = [
          'audio', 'histogram', 'image', 'scalar', 'text'
      ]
    self._checkBackwardsCompatibility(
        tf,
        golden_file_patterns,
        api_version,
        additional_private_map={'tf.compat': ['v1', 'v2']},
        omit_golden_symbols_map=omit_golden_symbols_map)
    self.assertTrue(api_version == 1 or not hasattr(tf, 'contrib'))
  def testAPIBackwardsCompatibilityV1(self):
    api_version = 1
    golden_file_patterns = os.path.join(
        resource_loader.get_root_dir_with_all_resources(),
        _KeyToFilePath('*', api_version))
    self._checkBackwardsCompatibility(
        tf.compat.v1,
        golden_file_patterns,
        api_version,
        additional_private_map={
            'tf': ['pywrap_tensorflow'],
            'tf.compat': ['v1', 'v2'],
        },
        omit_golden_symbols_map={'tensorflow': ['pywrap_tensorflow']})
  def testAPIBackwardsCompatibilityV2(self):
    api_version = 2
    golden_file_patterns = [
        os.path.join(resource_loader.get_root_dir_with_all_resources(),
                     _KeyToFilePath('*', api_version)),
        _GetTFNumpyGoldenPattern(api_version)]
    omit_golden_symbols_map = {}
    if FLAGS.only_test_core_api and not _TENSORBOARD_AVAILABLE:
      omit_golden_symbols_map['tensorflow.summary'] = [
          'audio', 'histogram', 'image', 'scalar', 'text'
      ]
    self._checkBackwardsCompatibility(
        tf.compat.v2,
        golden_file_patterns,
        api_version,
        additional_private_map={'tf.compat': ['v1', 'v2']},
        omit_golden_symbols_map=omit_golden_symbols_map)
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--update_goldens', type=bool, default=False, help=_UPDATE_GOLDENS_HELP)
  parser.add_argument(
      '--only_test_core_api',
      type=bool,
      help=_ONLY_TEST_CORE_API_HELP)
  parser.add_argument(
      '--verbose_diffs', type=bool, default=True, help=_VERBOSE_DIFFS_HELP)
  FLAGS, unparsed = parser.parse_known_args()
  _InitPathConstants()
  sys.argv = [sys.argv[0]] + unparsed
  test.main()
