
import gzip
import zlib
from six import BytesIO
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test
class DecodeCompressedOpTest(test.TestCase):
  def _compress(self, bytes_in, compression_type):
    if not compression_type:
      return bytes_in
    elif compression_type == "ZLIB":
      return zlib.compress(bytes_in)
    else:
      out = BytesIO()
      with gzip.GzipFile(fileobj=out, mode="wb") as f:
        f.write(bytes_in)
      return out.getvalue()
  def testDecompressShapeInference(self):
    with ops.Graph().as_default():
      for compression_type in ["ZLIB", "GZIP", ""]:
        with self.cached_session():
          in_bytes = array_ops.placeholder(dtypes.string, shape=[2])
          decompressed = parsing_ops.decode_compressed(
              in_bytes, compression_type=compression_type)
          self.assertEqual([2], decompressed.get_shape().as_list())
  def testDecompress(self):
    for compression_type in ["ZLIB", "GZIP", ""]:
      with self.cached_session():
        def decode(in_bytes, compression_type=compression_type):
          return parsing_ops.decode_compressed(
              in_bytes, compression_type=compression_type)
        in_val = [self._compress(b"AaAA", compression_type),
                  self._compress(b"bBbb", compression_type)]
        result = self.evaluate(decode(in_val))
        self.assertAllEqual([b"AaAA", b"bBbb"], result)
  def testDecompressWithRaw(self):
    for compression_type in ["ZLIB", "GZIP", ""]:
      with self.cached_session():
        def decode(in_bytes, compression_type=compression_type):
          decompressed = parsing_ops.decode_compressed(in_bytes,
                                                       compression_type)
          return parsing_ops.decode_raw(decompressed, out_type=dtypes.int16)
        result = self.evaluate(
            decode([self._compress(b"AaBC", compression_type)]))
        self.assertAllEqual(
            [[ord("A") + ord("a") * 256, ord("B") + ord("C") * 256]], result)
if __name__ == "__main__":
  test.main()
