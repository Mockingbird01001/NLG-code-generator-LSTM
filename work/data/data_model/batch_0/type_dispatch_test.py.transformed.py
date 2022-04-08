
from typing import Optional
from tensorflow.core.function.polymorphism import type_dispatch
from tensorflow.python.platform import test
from tensorflow.python.types import trace
class MockShape(trace.TraceType):
  def __init__(self, *shape: Optional[int]):
    self.shape = shape
  def is_subtype_of(self, other: "MockShape") ->bool:
    if len(self.shape) != len(other.shape):
      return False
    return all(o is None or s == o for s, o in zip(self.shape, other.shape))
  def most_specific_common_supertype(self, others):
    if any(len(other.shape) != len(self.shape) for other in others):
      return None
    dims = [
        dim if all(dim == other.shape[i]
                   for other in others) else None
        for i, dim in enumerate(self.shape)
    ]
    return MockShape(*dims)
  def __str__(self):
    return str(self.shape)
  def __repr__(self):
    return str(self)
  def __hash__(self) -> int:
    return hash(self.shape)
  def __eq__(self, other: "MockShape") -> bool:
    return self.shape == other.shape
class TypeDispatchTableTest(test.TestCase):
  def testVertical(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(None, None, None))
    table.add_target(MockShape(None, None, 1))
    table.add_target(MockShape(None, 1, 1))
    table.add_target(MockShape(1, 1, 1))
    self.assertEqual(
        list(table.targets), [
            MockShape(None, None, None),
            MockShape(None, None, 1),
            MockShape(None, 1, 1),
            MockShape(1, 1, 1)
        ])
  def testHorizontal(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(1,))
    table.add_target(MockShape(1, 2))
    table.add_target(MockShape(1, 2, 3))
    self.assertEqual(
        list(table.targets), [
            MockShape(1,),
            MockShape(1, 2),
            MockShape(1, 2, 3)
        ])
  def testDuplicateNodes(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(None, None))
    table.add_target(MockShape(1, None))
    table.add_target(MockShape(None, 2))
    table.add_target(MockShape(None, None))
    self.assertEqual(
        list(table.targets), [
            MockShape(None, None),
            MockShape(1, None),
            MockShape(None, 2)
        ])
  def testDeletion(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(None, None))
    table.add_target(MockShape(None, 1))
    table.add_target(MockShape(None, 2))
    self.assertEqual(
        list(table.targets), [
            MockShape(None, None),
            MockShape(None, 1),
            MockShape(None, 2)
        ])
    self.assertEqual(
        list(table.targets), [
            MockShape(None, None),
            MockShape(None, 1),
        ])
    self.assertEqual(
        list(table.targets), [
            MockShape(None, None),
            MockShape(None, 1),
        ])
  def testContains(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(None, None, None))
    table.add_target(MockShape(None, 1))
    table.add_target(MockShape(1, 1))
    table.add_target(MockShape(None, 2, 1))
    self.assertIn(MockShape(None, None, None), table.targets)
    self.assertIn(MockShape(None, 1), table.targets)
    self.assertIn(MockShape(1, 1), table.targets)
    self.assertIn(MockShape(None, 2, 1), table.targets)
    self.assertNotIn(MockShape(None, None, 1), table.targets)
    self.assertNotIn(MockShape(1, None), table.targets)
    self.assertNotIn(MockShape(1, 2), table.targets)
    self.assertNotIn(MockShape(None, 2, None), table.targets)
  def testDispatchExactMatches(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(None, None, None))
    table.add_target(MockShape(None, 1, None))
    table.add_target(MockShape(None, 1, 2))
    table.add_target(MockShape(None, 2, 2))
    self.assertEqual(
        table.dispatch(MockShape(None, 1, 2)), MockShape(None, 1, 2))
    self.assertEqual(
        table.dispatch(MockShape(None, 1, None)), MockShape(None, 1, None))
    self.assertEqual(
        table.dispatch(MockShape(None, None, None)),
        MockShape(None, None, None))
    self.assertEqual(
        table.dispatch(MockShape(None, 2, 2)), MockShape(None, 2, 2))
  def testDispatchMoreSpecific(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(None, None, None))
    table.add_target(MockShape(None, 1, None))
    table.add_target(MockShape(None, 1, 2))
    table.add_target(MockShape(None, 2, 2))
    self.assertEqual(table.dispatch(MockShape(1, 1, 2)), MockShape(None, 1, 2))
    self.assertEqual(
        table.dispatch(MockShape(1, 1, 3)), MockShape(None, 1, None))
    self.assertEqual(
        table.dispatch(MockShape(1, 3, 3)), MockShape(None, None, None))
    self.assertEqual(table.dispatch(MockShape(1, 2, 2)), MockShape(None, 2, 2))
  def testDispatchNoMatches(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(None, 1, None))
    table.add_target(MockShape(None, 1, 2))
    table.add_target(MockShape(None, 2, 2))
    self.assertIsNone(table.dispatch(MockShape(1, 2)))
    self.assertIsNone(table.dispatch(MockShape(1, 2, 3)))
    self.assertIsNone(table.dispatch(MockShape(1, 2, 3, 4)))
  def testDispatchCachedAddUpdates(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(None, None, None))
    self.assertEqual(
        table.dispatch(MockShape(1, 1, 2)), MockShape(None, None, None))
    table.add_target(MockShape(None, 1, None))
    self.assertEqual(
        table.dispatch(MockShape(1, 1, 2)), MockShape(None, 1, None))
    table.add_target(MockShape(None, 1, 2))
    self.assertEqual(table.dispatch(MockShape(1, 1, 2)), MockShape(None, 1, 2))
    table.add_target(MockShape(1, 1, 2))
    self.assertEqual(table.dispatch(MockShape(1, 1, 2)), MockShape(1, 1, 2))
  def testDispatchCachedDeleteUpdates(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(None, None, None))
    table.add_target(MockShape(None, 1, None))
    table.add_target(MockShape(None, 1, 2))
    table.add_target(MockShape(1, 1, 2))
    self.assertEqual(table.dispatch(MockShape(1, 1, 2)), MockShape(1, 1, 2))
    table.delete(MockShape(1, 1, 2))
    self.assertEqual(table.dispatch(MockShape(1, 1, 2)), MockShape(None, 1, 2))
    table.delete(MockShape(None, 1, 2))
    self.assertEqual(
        table.dispatch(MockShape(1, 1, 2)), MockShape(None, 1, None))
    table.delete(MockShape(None, 1, None))
    self.assertEqual(
        table.dispatch(MockShape(1, 1, 2)), MockShape(None, None, None))
  def testDispatchCacheOrderingDeterminism(self):
    table_1 = type_dispatch.TypeDispatchTable()
    table_1.add_target(MockShape(1, None, None))
    table_1.add_target(MockShape(None, 2, None))
    table_1.add_target(MockShape(None, None, 3))
    table_2 = type_dispatch.TypeDispatchTable()
    table_2.add_target(MockShape(None, 2, None))
    table_2.add_target(MockShape(1, None, None))
    table_2.add_target(MockShape(None, None, 3))
    table_3 = type_dispatch.TypeDispatchTable()
    table_3.add_target(MockShape(None, None, 3))
    table_3.add_target(MockShape(1, None, None))
    table_3.add_target(MockShape(None, 2, None))
    self.assertEqual(set(table_1.targets), set(table_2.targets))
    self.assertEqual(set(table_2.targets), set(table_3.targets))
    shape = MockShape(1, 2, 3)
    self.assertEqual(table_1.dispatch(shape), MockShape(1, None, None))
    self.assertEqual(table_2.dispatch(shape), MockShape(None, 2, None))
    self.assertEqual(table_3.dispatch(shape), MockShape(None, None, 3))
  def testGeneralizedExisting(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(None, None, None))
    table.add_target(MockShape(None, 1, None))
    table.add_target(MockShape(None, 1, 2))
    self.assertEqual(
        table.try_generalizing_trace_type(MockShape(None, 1, 3)),
        MockShape(None, None, None))
  def testGeneralizedNovel(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(None, 1, None))
    table.add_target(MockShape(None, 1, 2))
    self.assertEqual(
        table.try_generalizing_trace_type(MockShape(None, 2, 3)),
        MockShape(None, None, None))
  def testGeneralizedUnknown(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(None, 1))
    table.add_target(MockShape(None, 2))
    table.add_target(MockShape(None, 3))
    self.assertEqual(
        table.try_generalizing_trace_type(MockShape(None, 4, 3)),
        MockShape(None, 4, 3))
if __name__ == "__main__":
  test.main()
