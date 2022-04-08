
import portpicker
from tensorflow.python.eager import test
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.profiler import profiler_client
from tensorflow.python.profiler import profiler_v2 as profiler
class ProfilerClientTest(test_util.TensorFlowTestCase):
  def testTrace_ProfileIdleServer(self):
    test_port = portpicker.pick_unused_port()
    profiler.start_server(test_port)
    with self.assertRaises(errors.UnavailableError) as error:
      profiler_client.trace(
          'localhost:' + str(test_port), self.get_temp_dir(), duration_ms=10)
    self.assertStartsWith(str(error.exception), 'No trace event was collected')
  def testTrace_ProfileIdleServerWithOptions(self):
    test_port = portpicker.pick_unused_port()
    profiler.start_server(test_port)
    with self.assertRaises(errors.UnavailableError) as error:
      options = profiler.ProfilerOptions(
          host_tracer_level=3, device_tracer_level=0)
      profiler_client.trace(
          'localhost:' + str(test_port),
          self.get_temp_dir(),
          duration_ms=10,
          options=options)
    self.assertStartsWith(str(error.exception), 'No trace event was collected')
  def testMonitor_ProcessInvalidAddress(self):
    with self.assertRaises(errors.UnavailableError):
      profiler_client.monitor('localhost:6006', 2000)
if __name__ == '__main__':
  test.main()
