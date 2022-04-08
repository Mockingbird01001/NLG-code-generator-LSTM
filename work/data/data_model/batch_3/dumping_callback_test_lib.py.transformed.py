
import os
import shutil
import tempfile
import uuid
from tensorflow.python.debug.lib import check_numerics_callback
from tensorflow.python.debug.lib import debug_events_reader
from tensorflow.python.debug.lib import dumping_callback
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
class DumpingCallbackTestBase(test_util.TensorFlowTestCase):
  def setUp(self):
    super(DumpingCallbackTestBase, self).setUp()
    self.dump_root = tempfile.mkdtemp()
    self.tfdbg_run_id = str(uuid.uuid4())
  def tearDown(self):
    if os.path.isdir(self.dump_root):
      shutil.rmtree(self.dump_root, ignore_errors=True)
    check_numerics_callback.disable_check_numerics()
    dumping_callback.disable_dump_debug_info()
    super(DumpingCallbackTestBase, self).tearDown()
  def _readAndCheckMetadataFile(self):
    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      self.assertTrue(reader.tfdbg_run_id())
      self.assertEqual(reader.tensorflow_version(), versions.__version__)
      self.assertTrue(reader.tfdbg_file_version().startswith("debug.Event"))
