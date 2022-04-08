
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test
def _AddTest(test_class, op_name, testcase_name, fn):
  test_name = "_".join(["test", op_name, testcase_name])
  if hasattr(test_class, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test_class, test_name, fn)
class QrOpTest(test.TestCase):
  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testWrongDimensions(self):
    scalar = constant_op.constant(1.)
    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError),
                                "rank.* 2.*0"):
      linalg_ops.qr(scalar)
    vector = constant_op.constant([1., 2.])
    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError),
                                "rank.* 2.*1"):
      linalg_ops.qr(vector)
  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testConcurrentExecutesWithoutError(self):
    seed = [42, 24]
    all_ops = []
    for full_matrices_ in True, False:
      for rows_ in 4, 5:
        for cols_ in 4, 5:
          matrix_shape = [rows_, cols_]
          matrix1 = stateless_random_ops.stateless_random_normal(
              matrix_shape, seed)
          matrix2 = stateless_random_ops.stateless_random_normal(
              matrix_shape, seed)
          self.assertAllEqual(matrix1, matrix2)
          q1, r1 = linalg_ops.qr(matrix1, full_matrices=full_matrices_)
          q2, r2 = linalg_ops.qr(matrix2, full_matrices=full_matrices_)
          all_ops += [q1, q2, r1, r2]
    val = self.evaluate(all_ops)
    for i in range(0, len(val), 2):
      self.assertAllClose(val[i], val[i + 1])
def _GetQrOpTest(dtype_, shape_, full_matrices_, use_static_shape_):
  is_complex = dtype_ in (np.complex64, np.complex128)
  is_single = dtype_ in (np.float32, np.complex64)
  def CompareOrthogonal(self, x, y, rank):
    if is_single:
      atol = 5e-4
    else:
      atol = 5e-14
    x = x[..., 0:rank]
    y = y[..., 0:rank]
    sum_of_ratios = np.sum(np.divide(y, x), -2, keepdims=True)
    phases = np.divide(sum_of_ratios, np.abs(sum_of_ratios))
    x *= phases
    self.assertAllClose(x, y, atol=atol)
  def CheckApproximation(self, a, q, r):
    if is_single:
      tol = 1e-5
    else:
      tol = 1e-14
    a_recon = test_util.matmul_without_tf32(q, r)
    self.assertAllClose(a_recon, a, rtol=tol, atol=tol)
  def CheckUnitary(self, x):
    xx = test_util.matmul_without_tf32(x, x, adjoint_a=True)
    identity = array_ops.matrix_band_part(array_ops.ones_like(xx), 0, 0)
    if is_single:
      tol = 1e-5
    else:
      tol = 1e-14
    self.assertAllClose(identity, xx, atol=tol)
  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def Test(self):
    if not use_static_shape_ and context.executing_eagerly():
      return
    np.random.seed(1)
    x_np = np.random.uniform(
        low=-1.0, high=1.0, size=np.prod(shape_)).reshape(shape_).astype(dtype_)
    if is_complex:
      x_np += 1j * np.random.uniform(
          low=-1.0, high=1.0,
          size=np.prod(shape_)).reshape(shape_).astype(dtype_)
      if use_static_shape_:
        x_tf = constant_op.constant(x_np)
      else:
        x_tf = array_ops.placeholder(dtype_)
      q_tf, r_tf = linalg_ops.qr(x_tf, full_matrices=full_matrices_)
      if use_static_shape_:
        q_tf_val, r_tf_val = self.evaluate([q_tf, r_tf])
      else:
        with self.session() as sess:
          q_tf_val, r_tf_val = sess.run([q_tf, r_tf], feed_dict={x_tf: x_np})
      q_dims = q_tf_val.shape
      np_q = np.ndarray(q_dims, dtype_)
      np_q_reshape = np.reshape(np_q, (-1, q_dims[-2], q_dims[-1]))
      new_first_dim = np_q_reshape.shape[0]
      x_reshape = np.reshape(x_np, (-1, x_np.shape[-2], x_np.shape[-1]))
      for i in range(new_first_dim):
        if full_matrices_:
          np_q_reshape[i, :, :], _ = np.linalg.qr(
              x_reshape[i, :, :], mode="complete")
        else:
          np_q_reshape[i, :, :], _ = np.linalg.qr(
              x_reshape[i, :, :], mode="reduced")
      np_q = np.reshape(np_q_reshape, q_dims)
      CompareOrthogonal(self, np_q, q_tf_val, min(shape_[-2:]))
      CheckApproximation(self, x_np, q_tf_val, r_tf_val)
      CheckUnitary(self, q_tf_val)
  return Test
class QrGradOpTest(test.TestCase):
  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testNotImplementedCheck(self):
    np.random.seed(42)
    matrix = constant_op.constant(
        np.random.uniform(low=-1.0, high=1.0, size=(5, 2)).astype(np.float32))
    def _NoGrad(x):
      with backprop.GradientTape() as tape:
        tape.watch(x)
        ret = linalg_ops.qr(x, full_matrices=True)
      return tape.gradient(ret, x)
    m = r"QrGrad not implemented when nrows > ncols and full_matrices is true."
    with self.assertRaisesRegex(NotImplementedError, m):
      _NoGrad(matrix)
def _GetQrGradOpTest(dtype_, shape_, full_matrices_):
  def RandomInput():
    a = np.random.uniform(low=-1.0, high=1.0, size=shape_).astype(dtype_)
    if dtype_ in [np.complex64, np.complex128]:
      a += 1j * np.random.uniform(
          low=-1.0, high=1.0, size=shape_).astype(dtype_)
    return a
  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  @test_util.run_without_tensor_float_32("Tests Qr gradient, which calls matmul"
                                        )
  def Test(self):
    np.random.seed(42)
    epsilon = np.finfo(dtype_).eps
    delta = 0.1 * epsilon**(1.0 / 3.0)
    if dtype_ in [np.float32, np.complex64]:
      tol = 3e-2
    else:
      tol = 1e-6
    funcs = [
        lambda a: linalg_ops.qr(a, full_matrices=full_matrices_)[0],
        lambda a: linalg_ops.qr(a, full_matrices=full_matrices_)[1]
    ]
    for f in funcs:
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          f, [RandomInput()], delta=delta)
      self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)
  return Test
class QRBenchmark(test.Benchmark):
  shapes = [
      (4, 4),
      (8, 8),
      (16, 16),
      (101, 101),
      (256, 256),
      (1024, 1024),
      (2048, 2048),
      (1024, 2),
      (1024, 32),
      (1024, 128),
      (1024, 512),
      (1, 8, 8),
      (10, 8, 8),
      (100, 8, 8),
      (1, 256, 256),
      (10, 256, 256),
      (100, 256, 256),
  ]
  def benchmarkQROp(self):
    for shape_ in self.shapes:
      with ops.Graph().as_default(), \
          session.Session(config=benchmark.benchmark_config()) as sess, \
          ops.device("/cpu:0"):
        matrix_value = np.random.uniform(
            low=-1.0, high=1.0, size=shape_).astype(np.float32)
        matrix = variables.Variable(matrix_value)
        q, r = linalg_ops.qr(matrix)
        self.evaluate(variables.global_variables_initializer())
        self.run_op_benchmark(
            sess,
            control_flow_ops.group(q, r),
            min_iters=25,
            name="QR_cpu_{shape}".format(shape=shape_))
      if test.is_gpu_available(True):
        with ops.Graph().as_default(), \
            session.Session(config=benchmark.benchmark_config()) as sess, \
            ops.device("/device:GPU:0"):
          matrix_value = np.random.uniform(
              low=-1.0, high=1.0, size=shape_).astype(np.float32)
          matrix = variables.Variable(matrix_value)
          q, r = linalg_ops.qr(matrix)
          self.evaluate(variables.global_variables_initializer())
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(q, r),
              min_iters=25,
              name="QR_gpu_{shape}".format(shape=shape_))
if __name__ == "__main__":
  for dtype in np.float32, np.float64, np.complex64, np.complex128:
    for rows in 1, 2, 5, 10, 32, 100:
      for cols in 1, 2, 5, 10, 32, 100:
        for full_matrices in False, True:
          for batch_dims in [(), (3,)] + [(3, 2)] * (max(rows, cols) < 10):
            for use_static_shape in [True, False]:
              shape = batch_dims + (rows, cols)
              name = "%s_%s_full_%s_static_%s" % (dtype.__name__,
                                                  "_".join(map(str, shape)),
                                                  full_matrices,
                                                  use_static_shape)
              _AddTest(QrOpTest, "Qr", name,
                       _GetQrOpTest(dtype, shape, full_matrices,
                                    use_static_shape))
  for full_matrices in False, True:
    for dtype in np.float32, np.float64, np.complex64, np.complex128:
      for rows in 1, 2, 5, 10:
        for cols in 1, 2, 5, 10:
          if rows <= cols or (not full_matrices and rows > cols):
            for batch_dims in [(), (3,)] + [(3, 2)] * (max(rows, cols) < 10):
              shape = batch_dims + (rows, cols)
              name = "%s_%s_full_%s" % (dtype.__name__,
                                        "_".join(map(str, shape)),
                                        full_matrices)
              _AddTest(QrGradOpTest, "QrGrad", name,
                       _GetQrGradOpTest(dtype, shape, full_matrices))
  test.main()
