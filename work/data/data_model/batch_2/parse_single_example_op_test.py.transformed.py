
import itertools
import numpy as np
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
example = example_pb2.Example
feature = feature_pb2.Feature
features = lambda d: feature_pb2.Features(feature=d)
bytes_feature = lambda v: feature(bytes_list=feature_pb2.BytesList(value=v))
int64_feature = lambda v: feature(int64_list=feature_pb2.Int64List(value=v))
float_feature = lambda v: feature(float_list=feature_pb2.FloatList(value=v))
feature_list = lambda l: feature_pb2.FeatureList(feature=l)
feature_lists = lambda d: feature_pb2.FeatureLists(feature_list=d)
sequence_example = example_pb2.SequenceExample
def empty_sparse(dtype, shape=None):
  if shape is None:
    shape = [0]
  return (np.empty(shape=(0, len(shape)), dtype=np.int64),
          np.array([], dtype=dtype), np.array(shape, dtype=np.int64))
def flatten(list_of_lists):
  return itertools.chain.from_iterable(list_of_lists)
def flatten_values_tensors_or_sparse(tensors_list):
  return list(
      flatten([[v.indices, v.values, v.dense_shape] if isinstance(
          v, sparse_tensor.SparseTensor) else [v] for v in tensors_list]))
def _compare_output_to_expected(tester, dict_tensors, expected_tensors,
                                flat_output):
  tester.assertEqual(set(dict_tensors.keys()), set(expected_tensors.keys()))
  for k, v in dict_tensors.items():
    expected_v = expected_tensors[k]
    tf_logging.info("Comparing key: %s", k)
    if isinstance(v, sparse_tensor.SparseTensor):
      tester.assertEqual([k, len(expected_v)], [k, 3])
      tester.assertAllEqual(expected_v[0], flat_output[i])
      tester.assertAllEqual(expected_v[1], flat_output[i + 1])
      tester.assertAllEqual(expected_v[2], flat_output[i + 2])
      i += 3
    else:
      tester.assertAllEqual(expected_v, flat_output[i])
      i += 1
class ParseExampleTest(test.TestCase):
  def _test(self, kwargs, expected_values=None, expected_err=None):
    with self.cached_session() as sess:
      if expected_err:
        with self.assertRaisesWithPredicateMatch(expected_err[0],
                                                 expected_err[1]):
          out = parsing_ops.parse_single_example(**kwargs)
          sess.run(flatten_values_tensors_or_sparse(out.values()))
        return
      else:
        out = parsing_ops.parse_single_example(**kwargs)
        out_with_example_name = parsing_ops.parse_single_example(
            example_names="name", **kwargs)
        for result_dict in [out, out_with_example_name]:
          result = flatten_values_tensors_or_sparse(result_dict.values())
          tf_result = self.evaluate(result)
          _compare_output_to_expected(self, result_dict, expected_values,
                                      tf_result)
      for k, f in kwargs["features"].items():
        if isinstance(f, parsing_ops.FixedLenFeature) and f.shape is not None:
          self.assertEqual(tuple(out[k].get_shape().as_list()), f.shape)
        elif isinstance(f, parsing_ops.VarLenFeature):
          self.assertEqual(
              tuple(out[k].indices.get_shape().as_list()), (None, 1))
          self.assertEqual(tuple(out[k].values.get_shape().as_list()), (None,))
          self.assertEqual(
              tuple(out[k].dense_shape.get_shape().as_list()), (1,))
  @test_util.run_deprecated_v1
  def testEmptySerializedWithAllDefaults(self):
    sparse_name = "st_a"
    a_name = "a"
    b_name = "b"
    c_name = "c:has_a_tricky_name"
    a_default = [0, 42, 0]
    b_default = np.random.rand(3, 3).astype(bytes)
    c_default = np.random.rand(2).astype(np.float32)
    expected_output = {
        sparse_name: expected_st_a,
        a_name: np.array([a_default]),
        b_name: np.array(b_default),
        c_name: np.array(c_default),
    }
    self._test({
        "serialized": ops.convert_to_tensor(""),
        "features": {
            sparse_name:
                parsing_ops.VarLenFeature(dtypes.int64),
            a_name:
                parsing_ops.FixedLenFeature(
                    (1, 3), dtypes.int64, default_value=a_default),
            b_name:
                parsing_ops.FixedLenFeature(
                    (3, 3), dtypes.string, default_value=b_default),
            c_name:
                parsing_ops.FixedLenFeature(
                    (2,), dtypes.float32, default_value=c_default),
        }
    }, expected_output)
  def testEmptySerializedWithoutDefaultsShouldFail(self):
    input_features = {
        "st_a":
            parsing_ops.VarLenFeature(dtypes.int64),
        "a":
            parsing_ops.FixedLenFeature(
                (1, 3), dtypes.int64, default_value=[0, 42, 0]),
        "b":
            parsing_ops.FixedLenFeature(
                (3, 3),
                dtypes.string,
                default_value=np.random.rand(3, 3).astype(bytes)),
        "c":
            parsing_ops.FixedLenFeature(
                (2,), dtype=dtypes.float32),
    }
    original = example(features=features({"c": feature()}))
    self._test(
        {
            "serialized": original.SerializeToString(),
            "features": input_features,
        },
        expected_err=(errors_impl.OpError,
                      "Feature: c \\(data type: float\\) is required"))
    self._test(
        {
            "serialized": "",
            "features": input_features,
        },
        expected_err=(errors_impl.OpError,
                      "Feature: c \\(data type: float\\) is required"))
  def testDenseNotMatchingShapeShouldFail(self):
    original = example(features=features({
        "a": float_feature([-1, -1]),
    }))
    serialized = original.SerializeToString()
    self._test(
        {
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                "a": parsing_ops.FixedLenFeature((1, 3), dtypes.float32)
            }
        },
        expected_err=(errors_impl.OpError, "Key: a."))
  def testDenseDefaultNoShapeShouldFail(self):
    original = example(features=features({
        "a": float_feature([1, 1, 3]),
    }))
    serialized = original.SerializeToString()
    self._test(
        {
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                "a": parsing_ops.FixedLenFeature(None, dtypes.float32)
            }
        },
        expected_err=(ValueError, "Missing shape for feature a"))
  @test_util.run_deprecated_v1
  def testSerializedContainingSparse(self):
    original = [
        example(features=features({
            "st_c": float_feature([3, 4])
        })),
        example(features=features({
        })),
        example(features=features({
        })),
        example(features=features({
            "st_c": float_feature([1, 2, -1]),
            "st_d": bytes_feature([b"hi"])
        }))
    ]
    expected_outputs = [{
        "st_c": (np.array([[0], [1]], dtype=np.int64),
                 np.array([3.0, 4.0], dtype=np.float32),
                 np.array([2], dtype=np.int64)),
        "st_d":
            empty_sparse(bytes)
    }, {
        "st_c": empty_sparse(np.float32),
        "st_d": empty_sparse(bytes)
    }, {
        "st_c": empty_sparse(np.float32),
        "st_d": empty_sparse(bytes)
    }, {
        "st_c": (np.array([[0], [1], [2]], dtype=np.int64),
                 np.array([1.0, 2.0, -1.0], dtype=np.float32),
                 np.array([3], dtype=np.int64)),
        "st_d": (np.array([[0]], dtype=np.int64), np.array(["hi"], dtype=bytes),
                 np.array([1], dtype=np.int64))
    }]
    for proto, expected_output in zip(original, expected_outputs):
      self._test({
          "serialized": ops.convert_to_tensor(proto.SerializeToString()),
          "features": {
              "st_c": parsing_ops.VarLenFeature(dtypes.float32),
              "st_d": parsing_ops.VarLenFeature(dtypes.string)
          },
      }, expected_output)
  def testSerializedContainingSparseFeature(self):
    original = [
        example(features=features({
            "val": float_feature([3, 4]),
            "idx": int64_feature([5, 10])
        })),
        example(features=features({
            "idx": int64_feature([])
        })),
        example(features=features({
        })),
        example(features=features({
            "val": float_feature([1, 2, -1]),
            "idx":
        }))
    ]
    expected_outputs = [{
        "sp": (np.array([[5], [10]], dtype=np.int64),
               np.array([3.0, 4.0], dtype=np.float32),
               np.array([13], dtype=np.int64))
    }, {
        "sp": empty_sparse(np.float32, shape=[13])
    }, {
        "sp": empty_sparse(np.float32, shape=[13])
    }, {
        "sp": (np.array([[0], [3], [9]], dtype=np.int64),
               np.array([1.0, -1.0, 2.0], dtype=np.float32),
               np.array([13], dtype=np.int64))
    }]
    for proto, expected_output in zip(original, expected_outputs):
      self._test({
          "serialized": ops.convert_to_tensor(proto.SerializeToString()),
          "features": {
              "sp":
                  parsing_ops.SparseFeature(["idx"], "val", dtypes.float32,
                                            [13])
          }
      }, expected_output)
  def testSerializedContainingSparseFeatureReuse(self):
    original = [
        example(features=features({
            "val1": float_feature([3, 4]),
            "val2": float_feature([5, 6]),
            "idx": int64_feature([5, 10])
        })),
        example(features=features({
            "idx": int64_feature([])
        })),
    ]
    expected_outputs = [{
        "sp1": (np.array([[5], [10]], dtype=np.int64),
                np.array([3.0, 4.0], dtype=np.float32),
                np.array([13], dtype=np.int64)),
        "sp2": (np.array([[5], [10]], dtype=np.int64),
                np.array([5.0, 6.0], dtype=np.float32),
                np.array([7], dtype=np.int64))
    }, {
        "sp1": empty_sparse(np.float32, shape=[13]),
        "sp2": empty_sparse(np.float32, shape=[7])
    }]
    for proto, expected_output in zip(original, expected_outputs):
      self._test({
          "serialized": ops.convert_to_tensor(proto.SerializeToString()),
          "features": {
              "sp1":
                  parsing_ops.SparseFeature("idx", "val1", dtypes.float32, 13),
              "sp2":
                  parsing_ops.SparseFeature(
                      "idx",
                      "val2",
                      dtypes.float32,
                      size=7,
                      already_sorted=True)
          }
      }, expected_output)
  def testSerializedContaining3DSparseFeature(self):
    original = [
        example(features=features({
            "val": float_feature([3, 4]),
            "idx0": int64_feature([5, 10]),
            "idx1": int64_feature([0, 2]),
        })),
        example(features=features({
            "idx0": int64_feature([]),
            "idx1": int64_feature([]),
        })),
        example(features=features({
        })),
        example(features=features({
            "val": float_feature([1, 2, -1]),
            "idx1": int64_feature([1, 0, 2]),
        }))
    ]
    expected_outputs = [{
        "sp": (np.array([[5, 0], [10, 2]], dtype=np.int64),
               np.array([3.0, 4.0], dtype=np.float32),
               np.array([13, 3], dtype=np.int64))
    }, {
        "sp": empty_sparse(np.float32, shape=[13, 3])
    }, {
        "sp": empty_sparse(np.float32, shape=[13, 3])
    }, {
        "sp": (np.array([[0, 1], [3, 2], [9, 0]], dtype=np.int64),
               np.array([1.0, -1.0, 2.0], dtype=np.float32),
               np.array([13, 3], dtype=np.int64))
    }]
    for proto, expected_output in zip(original, expected_outputs):
      self._test({
          "serialized": ops.convert_to_tensor(proto.SerializeToString()),
          "features": {
              "sp":
                  parsing_ops.SparseFeature(["idx0", "idx1"], "val",
                                            dtypes.float32, [13, 3])
          }
      }, expected_output)
  def testSerializedContainingDense(self):
    aname = "a"
    bname = "b*has+a:tricky_name"
    original = [
        example(features=features({
            aname: float_feature([1, 1]),
            bname: bytes_feature([b"b0_str"]),
        })), example(features=features({
            aname: float_feature([-1, -1]),
            bname: bytes_feature([b""]),
        }))
    ]
    expected_outputs = [
        {
            aname:
                np.array([1, 1], dtype=np.float32).reshape(1, 2, 1),
            bname:
                np.array(["b0_str"], dtype=bytes).reshape(
                    1, 1, 1, 1)
        },
        {
            aname:
                np.array([-1, -1], dtype=np.float32).reshape(1, 2, 1),
            bname:
                np.array([""], dtype=bytes).reshape(
                    1, 1, 1, 1)
        }
    ]
    for proto, expected_output in zip(original, expected_outputs):
      self._test({
          "serialized": ops.convert_to_tensor(proto.SerializeToString()),
          "features": {
              aname:
                  parsing_ops.FixedLenFeature((1, 2, 1), dtype=dtypes.float32),
              bname:
                  parsing_ops.FixedLenFeature(
                      (1, 1, 1, 1), dtype=dtypes.string),
          }
      }, expected_output)
  def testSerializedContainingDenseWithConcat(self):
    aname = "a"
    bname = "b*has+a:tricky_name"
    original = [
        (example(features=features({
            aname: float_feature([10, 10]),
        })), example(features=features({
            aname: float_feature([1, 1]),
            bname: bytes_feature([b"b0_str"]),
        }))),
        (
            example(features=features({
                bname: bytes_feature([b"b100"]),
            })),
            example(features=features({
                aname: float_feature([-1, -1]),
                bname: bytes_feature([b"b1"]),
            })),),
    ]
    expected_outputs = [
        {
            aname:
                np.array([1, 1], dtype=np.float32).reshape(1, 2, 1),
            bname:
                np.array(["b0_str"], dtype=bytes).reshape(
                    1, 1, 1, 1)
        },
        {
            aname:
                np.array([-1, -1], dtype=np.float32).reshape(1, 2, 1),
            bname:
                np.array(["b1"], dtype=bytes).reshape(
                    1, 1, 1, 1)
        }
    ]
    for (m, n), expected_output in zip(original, expected_outputs):
      self._test({
          "serialized":
              ops.convert_to_tensor(
                  m.SerializeToString() + n.SerializeToString()),
          "features": {
              aname:
                  parsing_ops.FixedLenFeature((1, 2, 1), dtype=dtypes.float32),
              bname:
                  parsing_ops.FixedLenFeature(
                      (1, 1, 1, 1), dtype=dtypes.string),
          }
      }, expected_output)
  def testSerializedContainingDenseScalar(self):
    original = [
        example(features=features({
            "a": float_feature([1]),
        })), example(features=features({}))
    ]
    expected_outputs = [{
        "a": np.array([1], dtype=np.float32)
    }, {
        "a": np.array([-1], dtype=np.float32)
    }]
    for proto, expected_output in zip(original, expected_outputs):
      self._test({
          "serialized": ops.convert_to_tensor(proto.SerializeToString()),
          "features": {
              "a":
                  parsing_ops.FixedLenFeature(
                      (1,), dtype=dtypes.float32, default_value=-1),
          }
      }, expected_output)
  def testSerializedContainingDenseWithDefaults(self):
    original = [
        example(features=features({
            "a": float_feature([1, 1]),
        })),
        example(features=features({
            "b": bytes_feature([b"b1"]),
        })),
        example(features=features({
            "b": feature()
        })),
    ]
    expected_outputs = [
        {
            "a":
                np.array([1, 1], dtype=np.float32).reshape(1, 2, 1),
            "b":
                np.array("tmp_str", dtype=bytes).reshape(
                    1, 1, 1, 1)
        },
        {
            "a":
                np.array([3, -3], dtype=np.float32).reshape(1, 2, 1),
            "b":
                np.array("b1", dtype=bytes).reshape(
                    1, 1, 1, 1)
        },
        {
            "a":
                np.array([3, -3], dtype=np.float32).reshape(1, 2, 1),
            "b":
                np.array("tmp_str", dtype=bytes).reshape(
                    1, 1, 1, 1)
        }
    ]
    for proto, expected_output in zip(original, expected_outputs):
      self._test({
          "serialized": ops.convert_to_tensor(proto.SerializeToString()),
          "features": {
              "a":
                  parsing_ops.FixedLenFeature(
                      (1, 2, 1),
                      dtype=dtypes.float32,
                      default_value=[3.0, -3.0]),
              "b":
                  parsing_ops.FixedLenFeature(
                      (1, 1, 1, 1),
                      dtype=dtypes.string,
                      default_value="tmp_str"),
          }
      }, expected_output)
  @test_util.run_deprecated_v1
  def testSerializedContainingSparseAndSparseFeatureAndDenseWithNoDefault(self):
    original = [
        example(features=features({
            "c": float_feature([3, 4]),
            "val": bytes_feature([b"a", b"b"]),
            "idx": int64_feature([0, 3])
        })), example(features=features({
            "c": float_feature([1, 2]),
            "val": bytes_feature([b"c"]),
            "idx": int64_feature([7])
        }))
    ]
    a_default = np.array([[1, 2, 3]], dtype=np.int64)
    b_default = np.random.rand(3, 3).astype(bytes)
    expected_st_a = empty_sparse(np.int64)
    expected_outputs = [{
        "st_a":
            expected_st_a,
        "sp": (np.array([[0], [3]], dtype=np.int64),
               np.array(["a", "b"], dtype=bytes), np.array(
                   [13], dtype=np.int64)),
        "a":
            a_default,
        "b":
            b_default,
        "c":
            np.array([3, 4], dtype=np.float32)
    }, {
        "st_a":
            expected_st_a,
        "sp": (np.array([[7]], dtype=np.int64), np.array(["c"], dtype=bytes),
               np.array([13], dtype=np.int64)),
        "a":
            a_default,
        "b":
            b_default,
        "c":
            np.array([1, 2], dtype=np.float32)
    }]
    for proto, expected_output in zip(original, expected_outputs):
      self._test(
          {
              "serialized": ops.convert_to_tensor(proto.SerializeToString()),
              "features": {
                  "st_a":
                      parsing_ops.VarLenFeature(dtypes.int64),
                  "sp":
                      parsing_ops.SparseFeature("idx", "val", dtypes.string, 13
                                               ),
                  "a":
                      parsing_ops.FixedLenFeature(
                          (1, 3), dtypes.int64, default_value=a_default),
                  "b":
                      parsing_ops.FixedLenFeature(
                          (3, 3), dtypes.string, default_value=b_default),
                  "c":
                      parsing_ops.FixedLenFeature((2,), dtypes.float32),
              }
          },
          expected_output)
  @test_util.run_deprecated_v1
  def testSerializedContainingSparseAndSparseFeatureWithReuse(self):
    original = [
        example(features=features({
            "val": bytes_feature([b"a", b"b"]),
            "idx": int64_feature([0, 3])
        })), example(features=features({
            "val": bytes_feature([b"c", b"d"]),
            "idx": int64_feature([7, 1])
        }))
    ]
    expected_outputs = [{
        "idx": (np.array([[0], [1]], dtype=np.int64),
                np.array([0, 3], dtype=np.int64), np.array([2],
                                                           dtype=np.int64)),
        "sp": (np.array([[0], [3]], dtype=np.int64),
               np.array(["a", "b"], dtype=bytes), np.array(
                   [13], dtype=np.int64))
    },
                        {
                            "idx": (np.array([[0], [1]], dtype=np.int64),
                                    np.array([7, 1], dtype=np.int64),
                                    np.array([2], dtype=np.int64)),
                            "sp": (np.array([[1], [7]], dtype=np.int64),
                                   np.array(["d", "c"], dtype=bytes),
                                   np.array([13], dtype=np.int64))
                        }]
    for proto, expected_output in zip(original, expected_outputs):
      self._test({
          "serialized": ops.convert_to_tensor(proto.SerializeToString()),
          "features": {
              "idx":
                  parsing_ops.VarLenFeature(dtypes.int64),
              "sp":
                  parsing_ops.SparseFeature(["idx"], "val", dtypes.string, [13]
                                           ),
          }
      }, expected_output)
  @test_util.run_deprecated_v1
  def testSerializedContainingVarLenDense(self):
    aname = "a"
    bname = "b"
    cname = "c"
    dname = "d"
    original = [
        example(features=features({
            cname: int64_feature([2]),
        })),
        example(features=features({
            aname: float_feature([1, 1]),
            bname: bytes_feature([b"b0_str", b"b1_str"]),
        })),
        example(features=features({
            aname: float_feature([-1, -1, 2, 2]),
            bname: bytes_feature([b"b1"]),
        })),
        example(features=features({
            aname: float_feature([]),
            cname: int64_feature([3]),
        })),
    ]
    expected_outputs = [
        {
            aname: np.empty(shape=(0, 2, 1), dtype=np.int64),
            bname: np.empty(shape=(0, 1, 1, 1), dtype=bytes),
            cname: np.array([2], dtype=np.int64),
            dname: np.empty(shape=(0,), dtype=bytes)
        },
        {
            aname:
                np.array([[[1], [1]]], dtype=np.float32),
            bname:
                np.array(["b0_str", "b1_str"], dtype=bytes).reshape(2, 1, 1, 1),
            cname:
                np.empty(shape=(0,), dtype=np.int64),
            dname:
                np.empty(shape=(0,), dtype=bytes)
        },
        {
            aname: np.array([[[-1], [-1]], [[2], [2]]], dtype=np.float32),
            bname: np.array(["b1"], dtype=bytes).reshape(1, 1, 1, 1),
            cname: np.empty(shape=(0,), dtype=np.int64),
            dname: np.empty(shape=(0,), dtype=bytes)
        },
        {
            aname: np.empty(shape=(0, 2, 1), dtype=np.int64),
            bname: np.empty(shape=(0, 1, 1, 1), dtype=bytes),
            cname: np.array([3], dtype=np.int64),
            dname: np.empty(shape=(0,), dtype=bytes)
        },
    ]
    for proto, expected_output in zip(original, expected_outputs):
      self._test({
          "serialized": ops.convert_to_tensor(proto.SerializeToString()),
          "features": {
              aname:
                  parsing_ops.FixedLenSequenceFeature(
                      (2, 1), dtype=dtypes.float32, allow_missing=True),
              bname:
                  parsing_ops.FixedLenSequenceFeature(
                      (1, 1, 1), dtype=dtypes.string, allow_missing=True),
              cname:
                  parsing_ops.FixedLenSequenceFeature(
                      shape=[], dtype=dtypes.int64, allow_missing=True),
              dname:
                  parsing_ops.FixedLenSequenceFeature(
                      shape=[], dtype=dtypes.string, allow_missing=True),
          }
      }, expected_output)
    for proto, expected_output in zip(original, expected_outputs):
      self._test({
          "serialized": ops.convert_to_tensor(proto.SerializeToString()),
          "features": {
              aname:
                  parsing_ops.FixedLenSequenceFeature(
                      (2, 1), dtype=dtypes.float32, allow_missing=True),
              bname:
                  parsing_ops.FixedLenSequenceFeature(
                      (1, 1, 1), dtype=dtypes.string, allow_missing=True),
              cname:
                  parsing_ops.FixedLenSequenceFeature(
                      shape=[], dtype=dtypes.int64, allow_missing=True),
              dname:
                  parsing_ops.FixedLenSequenceFeature(
                      shape=[], dtype=dtypes.string, allow_missing=True),
          }
      }, expected_output)
    self._test(
        {
            "serialized":
                ops.convert_to_tensor(original[2].SerializeToString()),
            "features": {
                aname:
                    parsing_ops.FixedLenSequenceFeature(
                        (2, 1), dtype=dtypes.float32, allow_missing=True),
                bname:
                    parsing_ops.FixedLenSequenceFeature(
                        (2, 1, 1), dtype=dtypes.string, allow_missing=True),
            }
        },
        expected_err=(errors_impl.OpError, "Key: b."))
    self._test(
        {
            "serialized": ops.convert_to_tensor(""),
            "features": {
                aname:
                    parsing_ops.FixedLenSequenceFeature(
                        (2, 1),
                        dtype=dtypes.float32,
                        allow_missing=True,
                        default_value=[]),
                bname:
                    parsing_ops.FixedLenSequenceFeature(
                        (2, 1, 1), dtype=dtypes.string, allow_missing=True),
            }
        },
        expected_err=(ValueError,
                      "Cannot reshape a tensor with 0 elements to shape"))
    self._test(
        {
            "serialized": ops.convert_to_tensor(""),
            "features": {
                aname:
                    parsing_ops.FixedLenFeature(
                        (None, 2, 1), dtype=dtypes.float32),
                bname:
                    parsing_ops.FixedLenSequenceFeature(
                        (2, 1, 1), dtype=dtypes.string, allow_missing=True),
            }
        },
        expected_err=(ValueError,
                      "First dimension of shape for feature a unknown. "
                      "Consider using FixedLenSequenceFeature."))
    self._test(
        {
            "serialized": ops.convert_to_tensor(""),
            "features": {
                cname:
                    parsing_ops.FixedLenFeature(
                        (1, None), dtype=dtypes.int64, default_value=[[1]]),
            }
        },
        expected_err=(ValueError,
                      "All dimensions of shape for feature c need to be known "
                      r"but received \(1, None\)."))
    self._test(
        {
            "serialized": ops.convert_to_tensor(""),
            "features": {
                aname:
                    parsing_ops.FixedLenSequenceFeature(
                        (2, 1), dtype=dtypes.float32, allow_missing=True),
                bname:
                    parsing_ops.FixedLenSequenceFeature(
                        (1, 1, 1), dtype=dtypes.string, allow_missing=True),
                cname:
                    parsing_ops.FixedLenSequenceFeature(
                        shape=[], dtype=dtypes.int64, allow_missing=False),
                dname:
                    parsing_ops.FixedLenSequenceFeature(
                        shape=[], dtype=dtypes.string, allow_missing=True),
            }
        },
        expected_err=(ValueError,
                      "Unsupported: FixedLenSequenceFeature requires "
                      "allow_missing to be True."))
class ParseSingleExampleTest(test.TestCase):
  def _test(self, kwargs, expected_values=None, expected_err=None):
    with self.cached_session() as sess:
      if expected_err:
        with self.assertRaisesWithPredicateMatch(expected_err[0],
                                                 expected_err[1]):
          out = parsing_ops.parse_single_example(**kwargs)
          sess.run(flatten_values_tensors_or_sparse(out.values()))
        return
      else:
        out = parsing_ops.parse_single_example(**kwargs)
        tf_result = sess.run(flatten_values_tensors_or_sparse(out.values()))
        _compare_output_to_expected(self, out, expected_values, tf_result)
      for k, f in kwargs["features"].items():
        if isinstance(f, parsing_ops.FixedLenFeature) and f.shape is not None:
          self.assertEqual(tuple(out[k].get_shape()),
                           tensor_shape.as_shape(f.shape))
        elif isinstance(f, parsing_ops.VarLenFeature):
          self.assertEqual(
              tuple(out[k].indices.get_shape().as_list()), (None, 1))
          self.assertEqual(tuple(out[k].values.get_shape().as_list()), (None,))
          self.assertEqual(
              tuple(out[k].dense_shape.get_shape().as_list()), (1,))
  @test_util.run_deprecated_v1
  def testSingleExampleWithSparseAndSparseFeatureAndDense(self):
    original = example(features=features({
        "c": float_feature([3, 4]),
        "d": float_feature([0.0, 1.0]),
        "val": bytes_feature([b"a", b"b"]),
        "idx": int64_feature([0, 3]),
        "st_a": float_feature([3.0, 4.0])
    }))
    serialized = original.SerializeToString()
    expected_st_a = (
        np.array(
        np.array(
        np.array(
        np.array(
            [[0], [3]], dtype=np.int64), np.array(
                ["a", "b"], dtype="|S"), np.array(
    a_default = [1, 2, 3]
    b_default = np.random.rand(3, 3).astype(bytes)
    expected_output = {
        "st_a": expected_st_a,
        "sp": expected_sp,
        "a": [a_default],
        "b": b_default,
        "c": np.array([3, 4], dtype=np.float32),
        "d": np.array([0.0, 1.0], dtype=np.float32),
    }
    self._test(
        {
            "serialized":
                ops.convert_to_tensor(serialized),
            "features": {
                "st_a":
                    parsing_ops.VarLenFeature(dtypes.float32),
                "sp":
                    parsing_ops.SparseFeature(
                        ["idx"], "val", dtypes.string, [13]),
                "a":
                    parsing_ops.FixedLenFeature(
                        (1, 3), dtypes.int64, default_value=a_default),
                "b":
                    parsing_ops.FixedLenFeature(
                        (3, 3), dtypes.string, default_value=b_default),
                "c":
                    parsing_ops.FixedLenFeature(2, dtypes.float32),
                "d":
                    parsing_ops.FixedLenSequenceFeature([],
                                                        dtypes.float32,
                                                        allow_missing=True)
            }
        },
        expected_output)
  def testExampleLongerThanSpec(self):
    serialized = example(
        features=features({
            "a": bytes_feature([b"a", b"b"]),
        })).SerializeToString()
    self._test(
        {
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                "a": parsing_ops.FixedLenFeature(1, dtypes.string)
            }
        },
        expected_err=(errors_impl.OpError, "Can't parse serialized Example"))
if __name__ == "__main__":
  test.main()
