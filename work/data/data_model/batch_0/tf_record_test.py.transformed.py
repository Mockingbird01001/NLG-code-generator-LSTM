
import gzip
import os
import random
import string
import zlib
import six
from tensorflow.python.framework import errors_impl
from tensorflow.python.lib.io import tf_record
from tensorflow.python.platform import test
from tensorflow.python.util import compat
TFRecordCompressionType = tf_record.TFRecordCompressionType
_TEXT =
class TFCompressionTestCase(test.TestCase):
  def setUp(self):
    super(TFCompressionTestCase, self).setUp()
    self._num_files = 2
    self._num_records = 7
  def _Record(self, f, r):
    return compat.as_bytes("Record %d of file %d" % (r, f))
  def _CreateFiles(self, options=None, prefix=""):
    filenames = []
    for i in range(self._num_files):
      name = prefix + "tfrecord.%d.txt" % i
      records = [self._Record(i, j) for j in range(self._num_records)]
      fn = self._WriteRecordsToFile(records, name, options)
      filenames.append(fn)
    return filenames
  def _WriteRecordsToFile(self, records, name="tfrecord", options=None):
    fn = os.path.join(self.get_temp_dir(), name)
    with tf_record.TFRecordWriter(fn, options=options) as writer:
      for r in records:
        writer.write(r)
    return fn
  def _ZlibCompressFile(self, infile, name="tfrecord.z"):
    with open(infile, "rb") as f:
      cdata = zlib.compress(f.read())
    zfn = os.path.join(self.get_temp_dir(), name)
    with open(zfn, "wb") as f:
      f.write(cdata)
    return zfn
  def _GzipCompressFile(self, infile, name="tfrecord.gz"):
    with open(infile, "rb") as f:
      cdata = f.read()
    gzfn = os.path.join(self.get_temp_dir(), name)
    with gzip.GzipFile(gzfn, "wb") as f:
      f.write(cdata)
    return gzfn
  def _ZlibDecompressFile(self, infile, name="tfrecord"):
    with open(infile, "rb") as f:
      cdata = zlib.decompress(f.read())
    fn = os.path.join(self.get_temp_dir(), name)
    with open(fn, "wb") as f:
      f.write(cdata)
    return fn
  def _GzipDecompressFile(self, infile, name="tfrecord"):
    with gzip.GzipFile(infile, "rb") as f:
      cdata = f.read()
    fn = os.path.join(self.get_temp_dir(), name)
    with open(fn, "wb") as f:
      f.write(cdata)
    return fn
class TFRecordWriterTest(TFCompressionTestCase):
  def _AssertFilesEqual(self, a, b, equal):
    for an, bn in zip(a, b):
      with open(an, "rb") as af, open(bn, "rb") as bf:
        if equal:
          self.assertEqual(af.read(), bf.read())
        else:
          self.assertNotEqual(af.read(), bf.read())
  def _CompressionSizeDelta(self, records, options_a, options_b):
    fn_a = self._WriteRecordsToFile(records, "tfrecord_a", options=options_a)
    test_a = list(tf_record.tf_record_iterator(fn_a, options=options_a))
    self.assertEqual(records, test_a, options_a)
    fn_b = self._WriteRecordsToFile(records, "tfrecord_b", options=options_b)
    test_b = list(tf_record.tf_record_iterator(fn_b, options=options_b))
    self.assertEqual(records, test_b, options_b)
    return os.path.getsize(fn_a) - os.path.getsize(fn_b)
  def testWriteReadZLibFiles(self):
    options = tf_record.TFRecordOptions(TFRecordCompressionType.NONE)
    files = self._CreateFiles(options, prefix="uncompressed")
    zlib_files = [
        self._ZlibCompressFile(fn, "tfrecord_%s.z" % i)
        for i, fn in enumerate(files)
    ]
    self._AssertFilesEqual(files, zlib_files, False)
    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    compressed_files = self._CreateFiles(options, prefix="compressed")
    self._AssertFilesEqual(compressed_files, zlib_files, True)
    uncompressed_files = [
        self._ZlibDecompressFile(fn, "tfrecord_%s.z" % i)
        for i, fn in enumerate(compressed_files)
    ]
    self._AssertFilesEqual(uncompressed_files, files, True)
  def testWriteReadGzipFiles(self):
    options = tf_record.TFRecordOptions(TFRecordCompressionType.NONE)
    files = self._CreateFiles(options, prefix="uncompressed")
    gzip_files = [
        self._GzipCompressFile(fn, "tfrecord_%s.gz" % i)
        for i, fn in enumerate(files)
    ]
    self._AssertFilesEqual(files, gzip_files, False)
    options = tf_record.TFRecordOptions(TFRecordCompressionType.GZIP)
    compressed_files = self._CreateFiles(options, prefix="compressed")
    uncompressed_files = [
        self._GzipDecompressFile(fn, "tfrecord_%s.gz" % i)
        for i, fn in enumerate(compressed_files)
    ]
    self._AssertFilesEqual(uncompressed_files, files, True)
  def testNoCompressionType(self):
    self.assertEqual(
        "",
        tf_record.TFRecordOptions.get_compression_type_string(
            tf_record.TFRecordOptions()))
    self.assertEqual(
        "",
        tf_record.TFRecordOptions.get_compression_type_string(
            tf_record.TFRecordOptions("")))
    with self.assertRaises(ValueError):
      tf_record.TFRecordOptions(5)
    with self.assertRaises(ValueError):
      tf_record.TFRecordOptions("BZ2")
  def testZlibCompressionType(self):
    zlib_t = tf_record.TFRecordCompressionType.ZLIB
    self.assertEqual(
        "ZLIB",
        tf_record.TFRecordOptions.get_compression_type_string(
            tf_record.TFRecordOptions("ZLIB")))
    self.assertEqual(
        "ZLIB",
        tf_record.TFRecordOptions.get_compression_type_string(
            tf_record.TFRecordOptions(zlib_t)))
    self.assertEqual(
        "ZLIB",
        tf_record.TFRecordOptions.get_compression_type_string(
            tf_record.TFRecordOptions(tf_record.TFRecordOptions(zlib_t))))
  def testCompressionOptions(self):
    rnd = random.Random(123)
    random_record = compat.as_bytes(
        "".join(rnd.choice(string.digits) for _ in range(10000)))
    repeated_record = compat.as_bytes(_TEXT)
    for _ in range(10000):
      start_i = rnd.randint(0, len(_TEXT))
      length = rnd.randint(10, 200)
      repeated_record += _TEXT[start_i:start_i + length]
    records = [random_record, repeated_record, random_record]
    tests = [
    ]
    compression_type = tf_record.TFRecordCompressionType.ZLIB
    options_a = tf_record.TFRecordOptions(compression_type)
    for prop, value, delta_sign in tests:
      options_b = tf_record.TFRecordOptions(
          compression_type=compression_type, **{prop: value})
      delta = self._CompressionSizeDelta(records, options_a, options_b)
      self.assertTrue(
          delta == 0 if delta_sign == 0 else delta // delta_sign > 0,
          "Setting {} = {}, file was {} smaller didn't match sign of {}".format(
              prop, value, delta, delta_sign))
class TFRecordWriterZlibTest(TFCompressionTestCase):
  def testZLibFlushRecord(self):
    original = [b"small record"]
    fn = self._WriteRecordsToFile(original, "small_record")
    with open(fn, "rb") as h:
      buff = h.read()
    compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS)
    output = b""
    for c in buff:
      if isinstance(c, int):
        c = six.int2byte(c)
      output += compressor.compress(c)
      output += compressor.flush(zlib.Z_FULL_FLUSH)
    output += compressor.flush(zlib.Z_FULL_FLUSH)
    output += compressor.flush(zlib.Z_FULL_FLUSH)
    output += compressor.flush(zlib.Z_FINISH)
    with open(fn, "wb") as h:
      h.write(output)
    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    actual = list(tf_record.tf_record_iterator(fn, options=options))
    self.assertEqual(actual, original)
  def testZlibReadWrite(self):
    original = [b"foo", b"bar"]
    fn = self._WriteRecordsToFile(original, "zlib_read_write.tfrecord")
    zfn = self._ZlibCompressFile(fn, "zlib_read_write.tfrecord.z")
    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    actual = list(tf_record.tf_record_iterator(zfn, options=options))
    self.assertEqual(actual, original)
  def testZlibReadWriteLarge(self):
    original = [_TEXT * 10240]
    fn = self._WriteRecordsToFile(original, "zlib_read_write_large.tfrecord")
    zfn = self._ZlibCompressFile(fn, "zlib_read_write_large.tfrecord.z")
    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    actual = list(tf_record.tf_record_iterator(zfn, options=options))
    self.assertEqual(actual, original)
  def testGzipReadWrite(self):
    original = [b"foo", b"bar"]
    fn = self._WriteRecordsToFile(original, "gzip_read_write.tfrecord")
    gzfn = self._GzipCompressFile(fn, "tfrecord.gz")
    options = tf_record.TFRecordOptions(TFRecordCompressionType.GZIP)
    actual = list(tf_record.tf_record_iterator(gzfn, options=options))
    self.assertEqual(actual, original)
class TFRecordIteratorTest(TFCompressionTestCase):
  def setUp(self):
    super(TFRecordIteratorTest, self).setUp()
    self._num_records = 7
  def testIterator(self):
    records = [self._Record(0, i) for i in range(self._num_records)]
    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    fn = self._WriteRecordsToFile(records, "compressed_records", options)
    reader = tf_record.tf_record_iterator(fn, options)
    for expected in records:
      record = next(reader)
      self.assertEqual(expected, record)
    with self.assertRaises(StopIteration):
      record = next(reader)
  def testWriteZlibRead(self):
    original = [b"foo", b"bar"]
    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    fn = self._WriteRecordsToFile(original, "write_zlib_read.tfrecord.z",
                                  options)
    zfn = self._ZlibDecompressFile(fn, "write_zlib_read.tfrecord")
    actual = list(tf_record.tf_record_iterator(zfn))
    self.assertEqual(actual, original)
  def testWriteZlibReadLarge(self):
    original = [_TEXT * 10240]
    options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
    fn = self._WriteRecordsToFile(original, "write_zlib_read_large.tfrecord.z",
                                  options)
    zfn = self._ZlibDecompressFile(fn, "write_zlib_read_large.tfrecord")
    actual = list(tf_record.tf_record_iterator(zfn))
    self.assertEqual(actual, original)
  def testWriteGzipRead(self):
    original = [b"foo", b"bar"]
    options = tf_record.TFRecordOptions(TFRecordCompressionType.GZIP)
    fn = self._WriteRecordsToFile(original, "write_gzip_read.tfrecord.gz",
                                  options)
    gzfn = self._GzipDecompressFile(fn, "write_gzip_read.tfrecord")
    actual = list(tf_record.tf_record_iterator(gzfn))
    self.assertEqual(actual, original)
  def testReadGrowingFile_preservesReadOffset(self):
    """Verify that tf_record_iterator preserves read offset even after EOF.
    When a file is iterated to EOF, the iterator should raise StopIteration but
    not actually close the reader. Then if later new data is appended, the
    iterator should start returning that new data on the next call to next(),
    preserving the read offset. This behavior is required by TensorBoard.
    """
    fn = os.path.join(self.get_temp_dir(), "file.tfrecord")
    with tf_record.TFRecordWriter(fn) as writer:
      writer.write(b"one")
      writer.write(b"two")
      writer.flush()
      iterator = tf_record.tf_record_iterator(fn)
      self.assertEqual(b"one", next(iterator))
      self.assertEqual(b"two", next(iterator))
      with self.assertRaises(StopIteration):
        next(iterator)
      with self.assertRaises(StopIteration):
        next(iterator)
      writer.write(b"three")
      writer.flush()
      self.assertEqual(b"three", next(iterator))
      with self.assertRaises(StopIteration):
        next(iterator)
  def testReadTruncatedFile_preservesReadOffset(self):
    fn = os.path.join(self.get_temp_dir(), "temp_file")
    with tf_record.TFRecordWriter(fn) as writer:
      writer.write(b"truncated")
    with open(fn, "rb") as f:
      record_bytes = f.read()
    fn_truncated = os.path.join(self.get_temp_dir(), "truncated_file")
    with tf_record.TFRecordWriter(fn_truncated) as writer:
      writer.write(b"good")
    with open(fn_truncated, "ab", buffering=0) as f:
      f.write(record_bytes[:-1])
      iterator = tf_record.tf_record_iterator(fn_truncated)
      self.assertEqual(b"good", next(iterator))
      with self.assertRaises(errors_impl.DataLossError):
        next(iterator)
      with self.assertRaises(errors_impl.DataLossError):
        next(iterator)
      f.write(record_bytes[-1:])
      self.assertEqual(b"truncated", next(iterator))
      with self.assertRaises(StopIteration):
        next(iterator)
  def testReadReplacedFile_preservesReadOffset_afterReopen(self):
    """Verify that tf_record_iterator allows reopening at the same read offset.
    In some cases, data will be logically "appended" to a file by replacing the
    entire file with a new version that includes the additional data. For
    example, this can happen with certain GCS implementations (since GCS has no
    true append operation), or when using rsync without the `--inplace` option
    to transfer snapshots of a growing file. Since the iterator retains a handle
    to a stale version of the file, it won't return any of the new data.
    To force this to happen, callers can check for a replaced file (e.g. via a
    stat call that reflects an increased file size) and opt to close and reopen
    the iterator. When iteration is next attempted, this should result in
    reading from the newly opened file, while preserving the read offset. This
    behavior is required by TensorBoard.
    """
    def write_records_to_file(filename, records):
      writer = tf_record.TFRecordWriter(filename)
      for record in records:
        writer.write(record)
      writer.close()
    fn = os.path.join(self.get_temp_dir(), "orig_file")
    write_records_to_file(fn, [b"one", b"two"])
    iterator = tf_record.tf_record_iterator(fn)
    self.assertEqual(b"one", next(iterator))
    self.assertEqual(b"two", next(iterator))
    with self.assertRaises(StopIteration):
      next(iterator)
    with self.assertRaises(StopIteration):
      next(iterator)
    fn2 = os.path.join(self.get_temp_dir(), "new_file")
    write_records_to_file(fn2, [b"one", b"two", b"three"])
    if os.name == "nt":
      iterator.close()
    os.replace(fn2, fn)
    with self.assertRaises(StopIteration):
      next(iterator)
    with self.assertRaises(StopIteration):
      next(iterator)
    iterator.close()
    iterator.reopen()
    self.assertEqual(b"three", next(iterator))
    with self.assertRaises(StopIteration):
      next(iterator)
class TFRecordRandomReaderTest(TFCompressionTestCase):
  def testRandomReaderReadingWorks(self):
    records = [self._Record(0, i) for i in range(self._num_records)]
    fn = self._WriteRecordsToFile(records, "uncompressed_records")
    reader = tf_record.tf_record_random_reader(fn)
    offset = 0
    offsets = [offset]
    for i in range(self._num_records):
      record, offset = reader.read(offset)
      self.assertEqual(record, records[i])
      offsets.append(offset)
    with self.assertRaisesRegex(IndexError, r"Out of range.*offset"):
      reader.read(offset)
    for i in range(self._num_records - 1, 0, -1):
      record, offset = reader.read(offsets[i])
      self.assertEqual(offset, offsets[i + 1])
      self.assertEqual(record, records[i])
  def testRandomReaderThrowsErrorForInvalidOffset(self):
    records = [self._Record(0, i) for i in range(self._num_records)]
    fn = self._WriteRecordsToFile(records, "uncompressed_records")
    reader = tf_record.tf_record_random_reader(fn)
    with self.assertRaisesRegex(errors_impl.DataLossError, r"corrupted record"):
  def testClosingRandomReaderCausesErrorsForFurtherReading(self):
    records = [self._Record(0, i) for i in range(self._num_records)]
    fn = self._WriteRecordsToFile(records, "uncompressed_records")
    reader = tf_record.tf_record_random_reader(fn)
    reader.close()
    with self.assertRaisesRegex(errors_impl.FailedPreconditionError, r"closed"):
      reader.read(0)
class TFRecordWriterCloseAndFlushTests(test.TestCase):
  def setUp(self, compression_type=TFRecordCompressionType.NONE):
    super(TFRecordWriterCloseAndFlushTests, self).setUp()
    self._fn = os.path.join(self.get_temp_dir(), "tf_record_writer_test.txt")
    self._options = tf_record.TFRecordOptions(compression_type)
    self._writer = tf_record.TFRecordWriter(self._fn, self._options)
    self._num_records = 20
  def _Record(self, r):
    return compat.as_bytes("Record %d" % r)
  def testWriteAndLeaveOpen(self):
    records = list(map(self._Record, range(self._num_records)))
    for record in records:
      self._writer.write(record)
  def testWriteAndRead(self):
    records = list(map(self._Record, range(self._num_records)))
    for record in records:
      self._writer.write(record)
    self._writer.close()
    actual = list(tf_record.tf_record_iterator(self._fn, self._options))
    self.assertListEqual(actual, records)
  def testFlushAndRead(self):
    records = list(map(self._Record, range(self._num_records)))
    for record in records:
      self._writer.write(record)
    self._writer.flush()
    actual = list(tf_record.tf_record_iterator(self._fn, self._options))
    self.assertListEqual(actual, records)
  def testDoubleClose(self):
    self._writer.write(self._Record(0))
    self._writer.close()
    self._writer.close()
  def testFlushAfterCloseIsError(self):
    self._writer.write(self._Record(0))
    self._writer.close()
    with self.assertRaises(errors_impl.FailedPreconditionError):
      self._writer.flush()
  def testWriteAfterCloseIsError(self):
    self._writer.write(self._Record(0))
    self._writer.close()
    with self.assertRaises(errors_impl.FailedPreconditionError):
      self._writer.write(self._Record(1))
class TFRecordWriterCloseAndFlushGzipTests(TFRecordWriterCloseAndFlushTests):
  def setUp(self):
    super(TFRecordWriterCloseAndFlushGzipTests,
          self).setUp(TFRecordCompressionType.GZIP)
class TFRecordWriterCloseAndFlushZlibTests(TFRecordWriterCloseAndFlushTests):
  def setUp(self):
    super(TFRecordWriterCloseAndFlushZlibTests,
          self).setUp(TFRecordCompressionType.ZLIB)
if __name__ == "__main__":
  test.main()
