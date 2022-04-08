
"""Multipurpose TensorFlow Docker Helper.
- Assembles Dockerfiles
- Builds images (and optionally runs image tests)
- Pushes images to Docker Hub (provided with credentials)
Logs are written to stderr; the list of successfully built images is
written to stdout.
Read README.md (in this directory) for instructions!
"""
import collections
import copy
import errno
import itertools
import json
import multiprocessing
import os
import platform
import re
import shutil
import sys
from absl import app
from absl import flags
import cerberus
import docker
import yaml
FLAGS = flags.FLAGS
flags.DEFINE_string('hub_username', None,
                    'Dockerhub username, only used with --upload_to_hub')
flags.DEFINE_string(
    'hub_password', None,
    ('Dockerhub password, only used with --upload_to_hub. Use from an env param'
     ' so your password isn\'t in your history.'))
flags.DEFINE_integer('hub_timeout', 3600,
                     'Abort Hub upload if it takes longer than this.')
flags.DEFINE_string(
    'repository', 'tensorflow',
    'Tag local images as {repository}:tag (in addition to the '
    'hub_repository, if uploading to hub)')
flags.DEFINE_string(
    'hub_repository', None,
    'Push tags to this Docker Hub repository, e.g. tensorflow/tensorflow')
flags.DEFINE_boolean(
    'upload_to_hub',
    False,
    ('Push built images to Docker Hub (you must also provide --hub_username, '
     '--hub_password, and --hub_repository)'),
    short_name='u',
)
flags.DEFINE_boolean(
    'construct_dockerfiles', False, 'Do not build images', short_name='d')
flags.DEFINE_boolean(
    'keep_temp_dockerfiles',
    False,
    'Retain .temp.Dockerfiles created while building images.',
    short_name='k')
flags.DEFINE_boolean(
    'build_images', False, 'Do not build images', short_name='b')
flags.DEFINE_string(
    'run_tests_path', None,
    ('Execute test scripts on generated Dockerfiles before pushing them. '
     'Flag value must be a full path to the "tests" directory, which is usually'
     ' $(realpath ./tests). A failed tests counts the same as a failed build.'))
flags.DEFINE_boolean(
    'stop_on_failure', False,
    ('Stop processing tags if any one build fails. If False or not specified, '
     'failures are reported but do not affect the other images.'))
flags.DEFINE_boolean(
    'dry_run',
    False,
    'Do not build or deploy anything at all.',
    short_name='n',
)
flags.DEFINE_string(
    'exclude_tags_matching',
    None,
    ('Regular expression that skips processing on any tag it matches. Must '
     'match entire string, e.g. ".*gpu.*" ignores all GPU tags.'),
    short_name='x')
flags.DEFINE_string(
    'only_tags_matching',
    None,
    ('Regular expression that skips processing on any tag it does not match. '
     'Must match entire string, e.g. ".*gpu.*" includes only GPU tags.'),
    short_name='i')
flags.DEFINE_string(
    'dockerfile_dir',
    './dockerfiles', 'Path to an output directory for Dockerfiles.'
    ' Will be created if it doesn\'t exist.'
    ' Existing files in this directory will be deleted when new Dockerfiles'
    ' are made.',
    short_name='o')
flags.DEFINE_string(
    'partial_dir',
    './partials',
    'Path to a directory containing foo.partial.Dockerfile partial files.'
    ' can have subdirectories, e.g. "bar/baz.partial.Dockerfile".',
    short_name='p')
flags.DEFINE_multi_string(
    'release', [],
    'Set of releases to build and tag. Defaults to every release type.',
    short_name='r')
flags.DEFINE_multi_string(
    'arg', [],
    ('Extra build arguments. These are used for expanding tag names if needed '
     '(e.g. --arg _TAG_PREFIX=foo) and for using as build arguments (unused '
     'args will print a warning).'),
    short_name='a')
flags.DEFINE_boolean(
    'nocache', False,
    'Disable the Docker build cache; identical to "docker build --no-cache"')
flags.DEFINE_string(
    'spec_file',
    './spec.yml',
    'Path to the YAML specification file',
    short_name='s')
SCHEMA_TEXT =
class TfDockerTagValidator(cerberus.Validator):
  def __init__(self, *args, **kwargs):
    if 'partials' in kwargs:
      self.partials = kwargs['partials']
    super(cerberus.Validator, self).__init__(*args, **kwargs)
  def _validate_ispartial(self, ispartial, field, value):
    if ispartial and value not in self.partials:
      self._error(field,
                  '{} is not present in the partials directory.'.format(value))
  def _validate_isfullarg(self, isfullarg, field, value):
    if isfullarg and '=' not in value:
      self._error(field, '{} should be of the form ARG=VALUE.'.format(value))
    if not isfullarg and '=' in value:
      self._error(field, '{} should be of the form ARG (no =).'.format(value))
def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, flush=True, **kwargs)
def aggregate_all_slice_combinations(spec, slice_set_names):
  slice_sets = copy.deepcopy(spec['slice_sets'])
  for name in slice_set_names:
    for slice_set in slice_sets[name]:
      slice_set['set_name'] = name
  slices_grouped_but_not_keyed = [slice_sets[name] for name in slice_set_names]
  all_slice_combos = list(itertools.product(*slices_grouped_but_not_keyed))
  return all_slice_combos
def build_name_from_slices(format_string, slices, args, is_dockerfile=False):
  name_formatter = copy.deepcopy(args)
  name_formatter.update({s['set_name']: s['add_to_name'] for s in slices})
  name_formatter.update({
      s['set_name']: s['dockerfile_exclusive_name']
      for s in slices
      if is_dockerfile and 'dockerfile_exclusive_name' in s
  })
  name = format_string.format(**name_formatter)
  return name
def update_args_dict(args_dict, updater):
  if isinstance(updater, list):
    for arg in updater:
      key, sep, value = arg.partition('=')
      if sep == '=':
        args_dict[key] = value
  if isinstance(updater, dict):
    for key, value in updater.items():
      args_dict[key] = value
  return args_dict
def get_slice_sets_and_required_args(slice_sets, tag_spec):
  """Extract used-slice-sets and required CLI arguments from a spec string.
  For example, {FOO}{bar}{bat} finds FOO, bar, and bat. Assuming bar and bat
  are both named slice sets, FOO must be specified on the command line.
  Args:
     slice_sets: Dict of named slice sets
     tag_spec: The tag spec string, e.g. {_FOO}{blep}
  Returns:
     (used_slice_sets, required_args), a tuple of lists
  """
  required_args = []
  used_slice_sets = []
  extract_bracketed_words = re.compile(r'\{([^}]+)\}')
  possible_args_or_slice_set_names = extract_bracketed_words.findall(tag_spec)
  for name in possible_args_or_slice_set_names:
    if name in slice_sets:
      used_slice_sets.append(name)
    else:
      required_args.append(name)
  return (used_slice_sets, required_args)
def gather_tag_args(slices, cli_input_args, required_args):
  args = {}
  for s in slices:
    args = update_args_dict(args, s['args'])
  args = update_args_dict(args, cli_input_args)
  for arg in required_args:
    if arg not in args:
      eprint(('> Error: {} is not a valid slice_set, and also isn\'t an arg '
              'provided on the command line. If it is an arg, please specify '
              'it with --arg. If not, check the slice_sets list.'.format(arg)))
      exit(1)
  return args
def gather_slice_list_items(slices, key):
  return list(itertools.chain(*[s[key] for s in slices if key in s]))
def find_first_slice_value(slices, key):
  for s in slices:
    if key in s and s[key] is not None:
      return s[key]
  return None
def assemble_tags(spec, cli_args, enabled_releases, all_partials):
  tag_data = collections.defaultdict(list)
  for name, release in spec['releases'].items():
    for tag_spec in release['tag_specs']:
      if enabled_releases and name not in enabled_releases:
        eprint('> Skipping release {}'.format(name))
        continue
      used_slice_sets, required_cli_args = get_slice_sets_and_required_args(
          spec['slice_sets'], tag_spec)
      slice_combos = aggregate_all_slice_combinations(spec, used_slice_sets)
      for slices in slice_combos:
        tag_args = gather_tag_args(slices, cli_args, required_cli_args)
        tag_name = build_name_from_slices(tag_spec, slices, tag_args,
                                          release['is_dockerfiles'])
        used_partials = gather_slice_list_items(slices, 'partials')
        used_tests = gather_slice_list_items(slices, 'tests')
        test_runtime = find_first_slice_value(slices, 'test_runtime')
        dockerfile_subdirectory = find_first_slice_value(
            slices, 'dockerfile_subdirectory')
        dockerfile_contents = merge_partials(spec['header'], used_partials,
                                             all_partials)
        tag_data[tag_name].append({
            'release': name,
            'tag_spec': tag_spec,
            'is_dockerfiles': release['is_dockerfiles'],
            'upload_images': release['upload_images'],
            'cli_args': tag_args,
            'dockerfile_subdirectory': dockerfile_subdirectory or '',
            'partials': used_partials,
            'tests': used_tests,
            'test_runtime': test_runtime,
            'dockerfile_contents': dockerfile_contents,
        })
  return tag_data
def merge_partials(header, used_partials, all_partials):
  used_partials = list(used_partials)
  return '\n'.join([header] + [all_partials[u] for u in used_partials])
def upload_in_background(hub_repository, dock, image, tag):
  image.tag(hub_repository, tag=tag)
  print(dock.images.push(hub_repository, tag=tag))
def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise
def gather_existing_partials(partial_path):
  """Find and read all available partials.
  Args:
    partial_path (string): read partials from this directory.
  Returns:
    Dict[string, string] of partial short names (like "ubuntu/python" or
      "bazel") to the full contents of that partial.
  """
  partials = {}
  for path, _, files in os.walk(partial_path):
    for name in files:
      fullpath = os.path.join(path, name)
      if '.partial.Dockerfile' not in fullpath:
        eprint(('> Probably not a problem: skipping {}, which is not a '
                'partial.').format(fullpath))
        continue
      simple_name = fullpath[len(partial_path) + 1:-len('.partial.dockerfile')]
      with open(fullpath, 'r') as f:
        partial_contents = f.read()
      partials[simple_name] = partial_contents
  return partials
def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  with open(FLAGS.spec_file, 'r') as spec_file:
    tag_spec = yaml.safe_load(spec_file)
  partials = gather_existing_partials(FLAGS.partial_dir)
  schema = yaml.safe_load(SCHEMA_TEXT)
  v = TfDockerTagValidator(schema, partials=partials)
  if not v.validate(tag_spec):
    eprint('> Error: {} is an invalid spec! The errors are:'.format(
        FLAGS.spec_file))
    eprint(yaml.dump(v.errors, indent=2))
    exit(1)
  tag_spec = v.normalized(tag_spec)
  all_tags = assemble_tags(tag_spec, FLAGS.arg, FLAGS.release, partials)
  if FLAGS.construct_dockerfiles:
    eprint('> Emptying Dockerfile dir "{}"'.format(FLAGS.dockerfile_dir))
    shutil.rmtree(FLAGS.dockerfile_dir, ignore_errors=True)
    mkdir_p(FLAGS.dockerfile_dir)
  dock = docker.from_env()
  if FLAGS.upload_to_hub:
    if not FLAGS.hub_username:
      eprint('> Error: please set --hub_username when uploading to Dockerhub.')
      exit(1)
    if not FLAGS.hub_repository:
      eprint(
          '> Error: please set --hub_repository when uploading to Dockerhub.')
      exit(1)
    if not FLAGS.hub_password:
      eprint('> Error: please set --hub_password when uploading to Dockerhub.')
      exit(1)
    dock.login(
        username=FLAGS.hub_username,
        password=FLAGS.hub_password,
    )
  failed_tags = []
  succeeded_tags = []
  for tag, tag_defs in all_tags.items():
    for tag_def in tag_defs:
      eprint('> Working on {}'.format(tag))
      if FLAGS.exclude_tags_matching and re.match(FLAGS.exclude_tags_matching,
                                                  tag):
        eprint('>> Excluded due to match against "{}".'.format(
            FLAGS.exclude_tags_matching))
        continue
      if FLAGS.only_tags_matching and not re.match(FLAGS.only_tags_matching,
                                                   tag):
        eprint('>> Excluded due to failure to match against "{}".'.format(
            FLAGS.only_tags_matching))
        continue
      if FLAGS.construct_dockerfiles and tag_def['is_dockerfiles']:
        path = os.path.join(FLAGS.dockerfile_dir,
                            tag_def['dockerfile_subdirectory'],
                            tag + '.Dockerfile')
        eprint('>> Writing {}...'.format(path))
        if not FLAGS.dry_run:
          mkdir_p(os.path.dirname(path))
          with open(path, 'w') as f:
            f.write(tag_def['dockerfile_contents'])
      if not FLAGS.build_images:
        continue
      proc_arch = platform.processor()
      is_x86 = proc_arch.startswith('x86')
      if (is_x86 and any(arch in tag for arch in ['ppc64le']) or
          not is_x86 and proc_arch not in tag):
        continue
      dockerfile = os.path.join(FLAGS.dockerfile_dir, tag + '.temp.Dockerfile')
      if not FLAGS.dry_run:
        with open(dockerfile, 'w') as f:
          f.write(tag_def['dockerfile_contents'])
      eprint('>> (Temporary) writing {}...'.format(dockerfile))
      repo_tag = '{}:{}'.format(FLAGS.repository, tag)
      eprint('>> Building {} using build args:'.format(repo_tag))
      for arg, value in tag_def['cli_args'].items():
        eprint('>>> {}={}'.format(arg, value))
      tag_failed = False
      image, logs = None, []
      if not FLAGS.dry_run:
        try:
          resp = dock.api.build(
              timeout=FLAGS.hub_timeout,
              path='.',
              nocache=FLAGS.nocache,
              dockerfile=dockerfile,
              buildargs=tag_def['cli_args'],
              tag=repo_tag)
          last_event = None
          image_id = None
          while True:
            try:
              output = next(resp).decode('utf-8')
              json_output = json.loads(output.strip('\r\n'))
              if 'stream' in json_output:
                eprint(json_output['stream'], end='')
                match = re.search(r'(^Successfully built |sha256:)([0-9a-f]+)$',
                                  json_output['stream'])
                if match:
                  image_id = match.group(2)
                last_event = json_output['stream']
                logs.append(json_output)
            except StopIteration:
              eprint('Docker image build complete.')
              break
            except ValueError:
              eprint('Error parsing from docker image build: {}'.format(output))
          if image_id:
            image = dock.images.get(image_id)
          else:
            raise docker.errors.BuildError(last_event or 'Unknown', logs)
          if FLAGS.run_tests_path:
            if not tag_def['tests']:
              eprint('>>> No tests to run.')
            for test in tag_def['tests']:
              eprint('>> Testing {}...'.format(test))
              container, = dock.containers.run(
                  image,
                  '/tests/' + test,
                  working_dir='/',
                  log_config={'type': 'journald'},
                  detach=True,
                  stderr=True,
                  stdout=True,
                  volumes={
                      FLAGS.run_tests_path: {
                          'bind': '/tests',
                          'mode': 'ro'
                      }
                  },
                  runtime=tag_def['test_runtime']),
              ret = container.wait()
              code = ret['StatusCode']
              out = container.logs(stdout=True, stderr=False)
              err = container.logs(stdout=False, stderr=True)
              container.remove()
              if out:
                eprint('>>> Output stdout:')
                eprint(out.decode('utf-8'))
              else:
                eprint('>>> No test standard out.')
              if err:
                eprint('>>> Output stderr:')
                eprint(err.decode('utf-8'))
              else:
                eprint('>>> No test standard err.')
              if code != 0:
                eprint('>> {} failed tests with status: "{}"'.format(
                    repo_tag, code))
                failed_tags.append(tag)
                tag_failed = True
                if FLAGS.stop_on_failure:
                  eprint('>> ABORTING due to --stop_on_failure!')
                  exit(1)
              else:
                eprint('>> Tests look good!')
        except docker.errors.BuildError as e:
          eprint('>> {} failed to build with message: "{}"'.format(
              repo_tag, e.msg))
          eprint('>> Build logs follow:')
          log_lines = [l.get('stream', '') for l in e.build_log]
          eprint(''.join(log_lines))
          failed_tags.append(tag)
          tag_failed = True
          if FLAGS.stop_on_failure:
            eprint('>> ABORTING due to --stop_on_failure!')
            exit(1)
        if not FLAGS.keep_temp_dockerfiles:
          os.remove(dockerfile)
      if FLAGS.upload_to_hub:
        if not tag_def['upload_images']:
          continue
        if tag_failed:
          continue
        eprint('>> Uploading to {}:{}'.format(FLAGS.hub_repository, tag))
        if not FLAGS.dry_run:
          p = multiprocessing.Process(
              target=upload_in_background,
              args=(FLAGS.hub_repository, dock, image, tag))
          p.start()
      if not tag_failed:
        succeeded_tags.append(tag)
  if failed_tags:
    eprint(
        '> Some tags failed to build or failed testing, check scrollback for '
        'errors: {}'.format(','.join(failed_tags)))
    exit(1)
  eprint('> Writing built{} tags to standard out.'.format(
      ' and tested' if FLAGS.run_tests_path else ''))
  for tag in succeeded_tags:
    print('{}:{}'.format(FLAGS.repository, tag))
if __name__ == '__main__':
  app.run(main)
