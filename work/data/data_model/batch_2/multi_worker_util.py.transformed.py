
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.training import server_lib
def normalize_cluster_spec(cluster_spec):
  if isinstance(cluster_spec, (dict, cluster_pb2.ClusterDef)):
    return server_lib.ClusterSpec(cluster_spec)
  elif not isinstance(cluster_spec, server_lib.ClusterSpec):
    raise ValueError(
        "`cluster_spec' should be dict or a `tf.train.ClusterSpec` or a "
        "`tf.train.ClusterDef` object")
  return cluster_spec
def task_count(cluster_spec, task_type):
  try:
    return cluster_spec.num_tasks(task_type)
  except ValueError:
    return 0
def _validate_cluster_spec(cluster_spec,
                           task_type,
                           task_id):
  """Validates `cluster_spec`.
  It checks:
  1) task type is one of "chief", "worker", "ps", "evaluator", or not provided
     (None).
  2) whether there is such a task type as `task_type` in the `cluster_spec`. The
     only exception is `evaluator`. In other words, it is still a valid
     configuration when `task_type` is `evaluator` but it doesn't appear in
     `cluster_spec`. This is to be compatible with `TF_CONFIG` in Estimator.
  3) whether there is at most one "chief" job.
  4) whether there is at most one "evaluator" job.
  5) whether the `task_id` is smaller than the number of tasks for that
     particular `task_type`.
  Args:
    cluster_spec: a dict, `ClusterDef` or `ClusterSpec` object to be validated.
    task_type: string indicating the type of the task.
    task_id: the id of the `task_type` in this cluster.
  Raises:
    ValueError: if `cluster_spec` fails any check.
  """
  allowed_task_types = ("chief", "worker", "evaluator", "ps", None)
  cluster_spec = normalize_cluster_spec(cluster_spec)
  if any(job not in allowed_task_types for job in cluster_spec.jobs):
    raise ValueError("Disallowed task type found in cluster spec. Allowed "
                     "types are {} and the cluster spec is {}.".format(
                         allowed_task_types, cluster_spec))
  if task_type not in allowed_task_types:
    raise ValueError(
        "Unrecognized task_type: {}, valid task types are: {}".format(
            task_type, allowed_task_types))
  if (task_type and task_type not in cluster_spec.jobs and
      task_type != "evaluator"):
    raise ValueError("`task_type` %r not found in cluster_spec." % task_type)
  if task_count(cluster_spec, "chief") > 1:
    raise ValueError("There must be at most one 'chief' job.")
  if task_count(cluster_spec, "evaluator") > 1:
    raise ValueError("There must be at most one 'evaluator' job.")
  if task_type in cluster_spec.jobs and task_id >= task_count(
      cluster_spec, task_type):
    raise ValueError(
        "The `task_id` %d exceeds the maximum id of %s." % (task_id, task_type))
def is_chief(cluster_spec=None, task_type=None, task_id=None):
  if has_worker_context():
    return dc_context.get_current_worker_context().is_chief
  _validate_cluster_spec(cluster_spec, task_type, task_id)
  cluster_spec = normalize_cluster_spec(cluster_spec).as_dict()
  if task_type == "chief" or task_type == "evaluator":
    return True
  if ("chief" not in cluster_spec and task_type == "worker" and task_id == 0):
    return True
  return False
def collective_leader(cluster_spec, task_type, task_id):
  cluster_spec = normalize_cluster_spec(cluster_spec)
  if not cluster_spec.as_dict():
    return ""
  _validate_cluster_spec(cluster_spec, task_type, task_id)
  if task_type == "evaluator":
    return ""
  if "chief" in cluster_spec.jobs:
    return "/job:chief/replica:0/task:0"
  assert "worker" in cluster_spec.jobs
  return "/job:worker/replica:0/task:0"
def coordination_leader(cluster_spec):
  cluster_spec = normalize_cluster_spec(cluster_spec)
  if not cluster_spec.as_dict():
    return ""
  if "chief" in cluster_spec.jobs:
    return "/job:chief/replica:0/task:0"
  assert "worker" in cluster_spec.jobs
  return "/job:worker/replica:0/task:0"
def worker_count(cluster_spec, task_type):
  _validate_cluster_spec(cluster_spec, task_type, task_id=0)
  cluster_spec = normalize_cluster_spec(cluster_spec).as_dict()
  if task_type not in ["chief", "worker", "evaluator"]:
    raise ValueError("Unexpected `task_type` %r" % task_type)
  if task_type == "evaluator":
    return len(cluster_spec["evaluator"])
  else:
    return (len(cluster_spec.get("chief", [])) + len(
        cluster_spec.get("worker", [])))
def id_in_cluster(cluster_spec, task_type, task_id):
  """Returns a unique id for the task in the `task_type`'s cluster.
  It returns an id ranging from [0, `worker_count(task_type, task_id)`).
  Note: this function assumes that "evaluate" job is in its own cluster or its
  own partition of a cluster.
  Args:
    cluster_spec: a dict, `ClusterDef` or `ClusterSpec` object to be validated.
    task_type: string indicating the type of the task.
    task_id: the id of the `task_type` in this cluster.
  Returns:
    an int indicating the unique id.
  Throws:
    ValueError: if `task_type` is not "chief", "worker" or "evaluator".
  """
  _validate_cluster_spec(cluster_spec, task_type, task_id)
  cluster_spec = normalize_cluster_spec(cluster_spec).as_dict()
  if task_type == "chief":
    return 0
  if task_type == "worker":
    return task_id + len(cluster_spec.get("chief", []))
  if task_type == "evaluator":
    return task_id
  raise ValueError("There is no id for task_type %r" % task_type)
def should_save_checkpoint():
  """Returns whether the current worker should save checkpoints.
  In multi-worker training, if saving checkpoint is requested by user, or needed
  for fault-tolerance, the cluster should save checkpoint but not necessarily
  every worker in the cluster should.
  TODO(rchao): Consider generalizing this util to be `should_save_file` as there
  can be other files to save such as summary.
  Returns:
      Whether this particular worker in the cluster should save checkpoints.
  """
  return dc_context.get_current_worker_context().should_checkpoint
def should_load_checkpoint():
  return dc_context.get_current_worker_context().experimental_should_init
def wait_for_other_workers():
  return dc_context.get_current_worker_context().wait_for_other_workers()
def has_worker_context():
  return dc_context.get_current_worker_context() is not None
