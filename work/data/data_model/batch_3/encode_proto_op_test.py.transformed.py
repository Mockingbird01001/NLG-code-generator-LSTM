
from tensorflow.python.kernel_tests.proto import encode_proto_op_test_base as test_base
from tensorflow.python.ops import proto_ops
from tensorflow.python.platform import test
class EncodeProtoOpTest(test_base.EncodeProtoOpTestBase):
    super(EncodeProtoOpTest, self).__init__(encode_module=proto_ops,
                                            decode_module=proto_ops,
                                            methodName=methodName)
if __name__ == '__main__':
  test.main()
