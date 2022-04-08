
import traceback
from tensorflow.python.platform import test
from tensorflow.python.util import tf_stack
class TFStackTest(test.TestCase):
  def testFormatStackSelfConsistency(self):
    stacks = tf_stack.extract_stack(), traceback.extract_stack()
    self.assertEqual(
        traceback.format_list(stacks[0]), traceback.format_list(stacks[1]))
  def testFrameSummaryEquality(self):
    frames1 = tf_stack.extract_stack()
    frames2 = tf_stack.extract_stack()
    self.assertNotEqual(frames1[0], frames1[1])
    self.assertEqual(frames1[0], frames1[0])
    self.assertEqual(frames1[0], frames2[0])
  def testFrameSummaryEqualityAndHash(self):
    frame1, frame2 = tf_stack.extract_stack(), tf_stack.extract_stack()
    self.assertEqual(len(frame1), len(frame2))
    for f1, f2 in zip(frame1, frame2):
      self.assertEqual(f1, f2)
      self.assertEqual(hash(f1), hash(f1))
      self.assertEqual(hash(f1), hash(f2))
    self.assertEqual(frame1, frame2)
    self.assertEqual(hash(tuple(frame1)), hash(tuple(frame2)))
  def testLastUserFrame(self):
    frame = trace.last_user_frame()
  def testGetUserFrames(self):
    def func():
      frames = list(trace.get_user_frames())
      return frames
if __name__ == "__main__":
  test.main()
