
from tensorflow.core.grappler.costs import op_performance_data_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.grappler import _pywrap_tf_item as tf_item
class Item(object):
  def __init__(self,
               metagraph,
               ignore_colocation=True,
               ignore_user_placement=False):
    self._metagraph = metagraph
    self._item_graph = meta_graph_pb2.MetaGraphDef()
    self._item_graph.CopyFrom(metagraph)
    self._ignore_colocation = ignore_colocation
    self._ignore_user_placement = ignore_user_placement
    self._tf_item = None
    self._BuildTFItem()
  def IdentifyImportantOps(self, sort_topologically=False):
    return tf_item.TF_IdentifyImportantOps(self.tf_item, sort_topologically)
  def GetOpProperties(self):
    props = tf_item.TF_GetOpProperties(self.tf_item)
    properties = {}
    for key, values in props.items():
      prop = []
      for value in values:
        prop.append(
            op_performance_data_pb2.OpInfo.TensorProperties.FromString(value))
      properties[key] = prop
    return properties
  def GetColocationGroups(self):
    return tf_item.TF_GetColocationGroups(self.tf_item)
  @property
  def metagraph(self):
    return self._metagraph
  @property
  def tf_item(self):
    if self._item_graph != self._metagraph:
      self._BuildTFItem()
      self._item_graph.CopyFrom(self._metagraph)
    return self._tf_item
  def _BuildTFItem(self):
    self._tf_item = tf_item.TF_NewItem(self._metagraph.SerializeToString(),
                                       self._ignore_colocation,
                                       self._ignore_user_placement)
