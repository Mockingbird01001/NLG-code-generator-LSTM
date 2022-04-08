
from tensorflow.python.kernel_tests.proto import descriptor_source_test_base as test_base
from tensorflow.python.ops import proto_ops
from tensorflow.python.platform import test
class DescriptorSourceTest(test_base.DescriptorSourceTestBase):
    super(DescriptorSourceTest, self).__init__(decode_module=proto_ops,
                                               encode_module=proto_ops,
                                               methodName=methodName)
if __name__ == '__main__':
  test.main()
