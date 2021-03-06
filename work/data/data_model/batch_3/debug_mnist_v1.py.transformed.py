
"""Demo of the tfdbg curses CLI: Locating the source of bad numerical values.
The neural network in this demo is larged based on the tutorial at:
  tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
But modifications are made so that problematic numerical values (infs and nans)
appear in nodes of the graph during training.
"""
import argparse
import sys
import tempfile
import tensorflow
from tensorflow.python import debug as tf_debug
tf = tensorflow.compat.v1
IMAGE_SIZE = 28
HIDDEN_SIZE = 500
NUM_LABELS = 10
RAND_SEED = 42
FLAGS = None
def parse_args():
  """Parses commandline arguments.
  Returns:
    A tuple (parsed, unparsed) of the parsed object and a group of unparsed
      arguments that did not match the parser.
  """
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--max_steps",
      type=int,
      default=10,
      help="Number of steps to run trainer.")
  parser.add_argument(
      "--train_batch_size",
      type=int,
      default=100,
      help="Batch size used during training.")
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=0.025,
      help="Initial learning rate.")
  parser.add_argument(
      "--data_dir",
      type=str,
      default="/tmp/mnist_data",
      help="Directory for storing data")
  parser.add_argument(
      "--ui_type",
      type=str,
      default="curses",
      help="Command-line user interface type (curses | readline)")
  parser.add_argument(
      "--fake_data",
      type="bool",
      nargs="?",
      const=True,
      default=False,
      help="Use fake MNIST data for unit testing")
  parser.add_argument(
      "--debug",
      type="bool",
      nargs="?",
      const=True,
      default=False,
      help="Use debugger to track down bad values during training. "
      "Mutually exclusive with the --tensorboard_debug_address flag.")
  parser.add_argument(
      "--tensorboard_debug_address",
      type=str,
      default=None,
      help="Connect to the TensorBoard Debugger Plugin backend specified by "
      "the gRPC address (e.g., localhost:1234). Mutually exclusive with the "
      "--debug flag.")
  parser.add_argument(
      "--use_random_config_path",
      type="bool",
      nargs="?",
      const=True,
      default=False,
      help
)
  return parser.parse_known_args()
def main(_):
  if FLAGS.fake_data:
    imgs = tf.random.uniform(maxval=256, shape=(10, 28, 28), dtype=tf.int32)
    labels = tf.random.uniform(maxval=10, shape=(10,), dtype=tf.int32)
    mnist_train = imgs, labels
    mnist_test = imgs, labels
  else:
    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()
  def format_example(imgs, labels):
    imgs = tf.reshape(imgs, [-1, 28 * 28])
    imgs = tf.cast(imgs, tf.float32) / 255.0
    labels = tf.one_hot(labels, depth=10, dtype=tf.float32)
    return imgs, labels
  ds_train = tf.data.Dataset.from_tensor_slices(mnist_train)
  ds_train = ds_train.shuffle(
      1000, seed=RAND_SEED).repeat().batch(FLAGS.train_batch_size)
  ds_train = ds_train.map(format_example)
  it_train = ds_train.make_initializable_iterator()
  ds_test = tf.data.Dataset.from_tensors(mnist_test).repeat()
  ds_test = ds_test.map(format_example)
  it_test = ds_test.make_initializable_iterator()
  sess = tf.InteractiveSession()
  with tf.name_scope("input"):
    handle = tf.placeholder(tf.string, shape=())
    iterator = tf.data.Iterator.from_string_handle(
        handle, (tf.float32, tf.float32),
        ((None, IMAGE_SIZE * IMAGE_SIZE), (None, 10)))
    x, y_ = iterator.get_next()
  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=RAND_SEED)
    return tf.Variable(initial)
  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
      with tf.name_scope("weights"):
        weights = weight_variable([input_dim, output_dim])
      with tf.name_scope("biases"):
        biases = bias_variable([output_dim])
      with tf.name_scope("Wx_plus_b"):
        preactivate = tf.matmul(input_tensor, weights) + biases
      activations = act(preactivate)
      return activations
  hidden = nn_layer(x, IMAGE_SIZE**2, HIDDEN_SIZE, "hidden")
  logits = nn_layer(hidden, HIDDEN_SIZE, NUM_LABELS, "output", tf.identity)
  y = tf.nn.softmax(logits)
  with tf.name_scope("cross_entropy"):
    diff = -(y_ * tf.log(y))
    with tf.name_scope("total"):
      cross_entropy = tf.reduce_mean(diff)
  with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(
        FLAGS.learning_rate).minimize(cross_entropy)
  with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope("accuracy"):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  sess.run(tf.global_variables_initializer())
  sess.run(it_train.initializer)
  sess.run(it_test.initializer)
  train_handle = sess.run(it_train.string_handle())
  test_handle = sess.run(it_test.string_handle())
  if FLAGS.debug and FLAGS.tensorboard_debug_address:
    raise ValueError(
        "The --debug and --tensorboard_debug_address flags are mutually "
        "exclusive.")
  if FLAGS.debug:
    if FLAGS.use_random_config_path:
      _, config_file_path = tempfile.mkstemp(".tfdbg_config")
    else:
      config_file_path = None
    sess = tf_debug.LocalCLIDebugWrapperSession(
        sess, ui_type=FLAGS.ui_type, config_file_path=config_file_path)
  elif FLAGS.tensorboard_debug_address:
    sess = tf_debug.TensorBoardDebugWrapperSession(
        sess, FLAGS.tensorboard_debug_address)
  for i in range(FLAGS.max_steps):
    acc = sess.run(accuracy, feed_dict={handle: test_handle})
    print("Accuracy at step %d: %s" % (i, acc))
    sess.run(train_step, feed_dict={handle: train_handle})
if __name__ == "__main__":
  FLAGS, unparsed = parse_args()
  with tf.Graph().as_default():
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
