
from tensorflow.python.grappler import _pywrap_cost_analyzer as tf_wrap
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import item as gitem
def GenerateCostReport(metagraph,
                       per_node_report=False,
                       verbose=False,
                       cluster=None):
  if cluster is None:
    cluster = gcluster.Cluster(disable_detailed_stats=False)
  return tf_wrap.GenerateCostReport(metagraph.SerializeToString(),
                                    per_node_report, verbose,
                                    cluster.tf_cluster)
def GenerateMemoryReport(metagraph, detailed_report=True, cluster=None):
  if cluster is None:
    cluster = gcluster.Cluster(
        disable_detailed_stats=True, disable_timeline=True)
  item = gitem.Item(metagraph)
  peak_usage = cluster.DeterminePeakMemoryUsage(item)
  report = ""
  for device, snapshot in peak_usage.items():
    peak_usage = snapshot[0]
    report += "Peak usage for device " + device + ": " + str(
        peak_usage) + " bytes\n"
    if detailed_report:
      live_tensors = snapshot[1]
      for tensor in live_tensors:
        op_name = tensor[0]
        output_id = tensor[1]
        mem_used = tensor[2]
        report += "  " + str(op_name) + ":" + str(output_id) + " uses " + str(
            mem_used) + " bytes\n"
  return report
