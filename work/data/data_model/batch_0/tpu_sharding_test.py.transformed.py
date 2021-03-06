
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_sharding
class ShardingTest(test.TestCase):
  def testFreeze(self):
    p1 = tpu_sharding.ShardingPolicy()
    p1.freeze()
    self.assertEqual(p1.number_of_shards,
                     tpu_sharding._DEFAULT_NUMBER_OF_SHARDS)
    self.assertEqual(p1.shard_dimension, tpu_sharding._DEFAULT_SHARD_DIMENSION)
    p2 = tpu_sharding.ShardingPolicy()
    p2.set_number_of_shards(17)
    p2.set_shard_dimension(23)
    p2.freeze()
    self.assertEqual(p2.number_of_shards, 17)
    self.assertEqual(p2.shard_dimension, 23)
  def testFrozen(self):
    p1 = tpu_sharding.ShardingPolicy()
    p1.freeze()
    with self.assertRaises(ValueError):
      p1.set_number_of_shards(17)
    with self.assertRaises(ValueError):
      p1.set_shard_dimension(22)
  def testStr(self):
    p1 = tpu_sharding.ShardingPolicy()
    self.assertEqual(str(p1), "ShardingPolicy(unset)")
    p1.set_number_of_shards(17)
    self.assertEqual(str(p1), "ShardingPolicy(unset)")
    p1.set_shard_dimension(8)
    self.assertEqual(str(p1), "ShardingPolicy(17 shards dimension 8)")
  def testMerge(self):
    p1 = tpu_sharding.ShardingPolicy()
    p1.set_number_of_shards(17)
    p1.set_shard_dimension(23)
    p2 = tpu_sharding.ShardingPolicy()
    p2.merge(p1)
    self.assertEqual(p2.number_of_shards, 17)
    self.assertEqual(p2.shard_dimension, 23)
    p1 = tpu_sharding.ShardingPolicy()
    p1.set_shard_dimension(12)
    p2.merge(p1)
    self.assertEqual(p2.number_of_shards, 17)
    self.assertEqual(p2.shard_dimension, 12)
    p2.freeze()
    p2.merge(p1)
    self.assertEqual(p2.number_of_shards, 17)
    self.assertEqual(p2.shard_dimension, 12)
    p1.set_number_of_shards(1)
    with self.assertRaises(ValueError):
      p2.merge(p1)
    p1 = tpu_sharding.ShardingPolicy()
    p1.set_number_of_shards(17)
    p2.merge(p1)
    p1.set_shard_dimension(2)
    with self.assertRaises(ValueError):
      p2.merge(p1)
  def testGetShardedShape(self):
    p = tpu_sharding.ShardingPolicy()
    p.set_number_of_shards(3)
    p.set_shard_dimension(1)
    self.assertEqual(p.get_sharded_shape([4, 9]), [4, 3])
    p.freeze()
    with self.assertRaises(ValueError):
      p.set_shard_dimension(0)
    with self.assertRaises(ValueError):
      _ = p.get_sharded_shape([4, 9], shard_index=4)
    with self.assertRaises(ValueError):
      _ = p.get_sharded_shape([4, 9], shard_index=-1)
    with self.assertRaises(TypeError):
      _ = p.get_sharded_shape("not_a_shape")
    with self.assertRaises(ValueError):
      _ = p.get_sharded_shape(tensor_shape.TensorShape(None))
    with self.assertRaises(ValueError):
      _ = p.get_sharded_shape([4, 10], shard_index=-1)
  def testGetUnpartitionedShape(self):
    p = tpu_sharding.ShardingPolicy()
    p.set_number_of_shards(3)
    p.set_shard_dimension(1)
    p.set_number_of_partitions(4)
    self.assertEqual(p.get_unpartitioned_shape([3, 5]), [3, 20])
    p.freeze()
    with self.assertRaises(ValueError):
      _ = p.get_unpartitioned_shape([3, None])
  def testGetUnshardedShape(self):
    p = tpu_sharding.ShardingPolicy()
    p.set_number_of_shards(2)
    p.set_shard_dimension(1)
    self.assertEqual(p.get_unsharded_shape([[4, 3], [4, 3]]), [4, 6])
    with self.assertRaises(ValueError):
      _ = p.get_unsharded_shape([[4, 3]])
    with self.assertRaises(ValueError):
      _ = p.get_unsharded_shape([[4, 3], [4, 3], [4, 3]])
    with self.assertRaises(ValueError):
      _ = p.get_unsharded_shape([[4, 3], [4, 2]])
    with self.assertRaises(TypeError):
      _ = p.get_unsharded_shape([[4, 3], "not_a_shape"])
    with self.assertRaises(ValueError):
      _ = p.get_unsharded_shape([None, [4, 3]])
    with self.assertRaises(ValueError):
      _ = p.get_unsharded_shape([[2], [4, 3]])
  def testScalar(self):
    p = tpu_sharding.ShardingPolicy()
    p.freeze()
    self.assertEqual(p.get_sharded_shape([]), [])
    self.assertEqual(p.get_unsharded_shape([[]]), [])
if __name__ == "__main__":
  test.main()
