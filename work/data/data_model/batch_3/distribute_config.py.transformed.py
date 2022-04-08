
import collections
class DistributeConfig(
    collections.namedtuple(
        'DistributeConfig',
        ['train_distribute', 'eval_distribute', 'remote_cluster'])):
  def __new__(cls,
              train_distribute=None,
              eval_distribute=None,
              remote_cluster=None):
    return super(DistributeConfig, cls).__new__(cls, train_distribute,
                                                eval_distribute, remote_cluster)
