
def collect_function_arg_names(function_names):
  function_to_args = {}
  def visit(unused_path, unused_parent, children):
    for child in children:
      _, attr = tf_decorator.unwrap(child[1])
      api_names_v1 = tf_export.get_v1_names(attr)
      api_names_v1 = ['tf.%s' % name for name in api_names_v1]
      matches_function_names = any(
          name in function_names for name in api_names_v1)
      if matches_function_names:
        if tf_inspect.isclass(attr):
          arg_list = tf_inspect.getargspec(
              getattr(attr, '__init__'))[0]
        else:
          arg_list = tf_inspect.getargspec(attr)[0]
        for name in api_names_v1:
          function_to_args[name] = arg_list
  visitor = public_api.PublicAPIVisitor(visit)
  visitor.do_not_descend_map['tf'].append('contrib')
  visitor.do_not_descend_map['tf.compat'] = ['v1', 'v2']
  traverse.traverse(tf, visitor)
  return function_to_args
def get_reorder_line(name, arg_list):
  return '    \'%s\': %s' % (name, str(arg_list))
def update_reorders_v2(output_file_path):
  reordered_function_names = (
      tf_upgrade_v2.TFAPIChangeSpec().reordered_function_names)
  all_reorders = collect_function_arg_names(reordered_function_names)
  rename_lines = [
      get_reorder_line(name, arg_names)
      for name, arg_names in all_reorders.items()]
  renames_file_text = '%sreorders = {\n%s\n}\n' % (
      _FILE_HEADER, ',\n'.join(sorted(rename_lines)))
  file_io.write_string_to_file(output_file_path, renames_file_text)
def main(unused_argv):
  update_reorders_v2(_OUTPUT_FILE_PATH)
if __name__ == '__main__':
  app.run(main=main)
