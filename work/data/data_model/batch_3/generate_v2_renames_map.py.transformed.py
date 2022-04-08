
def get_canonical_name(v2_names, v1_name):
  if v2_names:
    return v2_names[0]
  return 'compat.v1.%s' % v1_name
def get_all_v2_names():
  def visit(unused_path, unused_parent, children):
    for child in children:
      _, attr = tf_decorator.unwrap(child[1])
      api_names_v2 = tf_export.get_v2_names(attr)
      for name in api_names_v2:
        v2_names.add(name)
  visitor = public_api.PublicAPIVisitor(visit)
  visitor.do_not_descend_map['tf'].append('contrib')
  visitor.do_not_descend_map['tf.compat'] = ['v1']
  traverse.traverse(tf.compat.v2, visitor)
  return v2_names
def collect_constant_renames():
  """Looks for constants that need to be renamed in TF 2.0.
  Returns:
    Set of tuples of the form (current name, new name).
  """
  renames = set()
  for module in sys.modules.values():
    constants_v1_list = tf_export.get_v1_constants(module)
    constants_v2_list = tf_export.get_v2_constants(module)
    constants_v1 = {constant_name: api_names
                    for api_names, constant_name in constants_v1_list}
    constants_v2 = {constant_name: api_names
                    for api_names, constant_name in constants_v2_list}
    for constant_name, api_names_v1 in constants_v1.items():
      api_names_v2 = constants_v2[constant_name]
      for name in api_names_v1:
        if name not in api_names_v2:
          renames.add((name, get_canonical_name(api_names_v2, name)))
  return renames
def collect_function_renames():
  """Looks for functions/classes that need to be renamed in TF 2.0.
  Returns:
    Set of tuples of the form (current name, new name).
  """
  renames = set()
  def visit(unused_path, unused_parent, children):
    for child in children:
      _, attr = tf_decorator.unwrap(child[1])
      api_names_v1 = tf_export.get_v1_names(attr)
      api_names_v2 = tf_export.get_v2_names(attr)
      deprecated_api_names = set(api_names_v1) - set(api_names_v2)
      for name in deprecated_api_names:
        renames.add((name, get_canonical_name(api_names_v2, name)))
  visitor = public_api.PublicAPIVisitor(visit)
  visitor.do_not_descend_map['tf'].append('contrib')
  visitor.do_not_descend_map['tf.compat'] = ['v1', 'v2']
  traverse.traverse(tf, visitor)
  v2_names = get_all_v2_names()
  renames = set((name, new_name) for name, new_name in renames
                if name not in v2_names)
  return renames
def get_rename_line(name, canonical_name):
  return '    \'tf.%s\': \'tf.%s\'' % (name, canonical_name)
def update_renames_v2(output_file_path):
  function_renames = collect_function_renames()
  constant_renames = collect_constant_renames()
  all_renames = function_renames.union(constant_renames)
  manual_renames = set(
      all_renames_v2.manual_symbol_renames.keys())
  rename_lines = [
      get_rename_line(name, canonical_name)
      for name, canonical_name in all_renames
      if 'tf.' + six.ensure_str(name) not in manual_renames
  ]
  renames_file_text = '%srenames = {\n%s\n}\n' % (
      _FILE_HEADER, ',\n'.join(sorted(rename_lines)))
  file_io.write_string_to_file(output_file_path, renames_file_text)
def main(unused_argv):
  update_renames_v2(_OUTPUT_FILE_PATH)
if __name__ == '__main__':
  app.run(main=main)
