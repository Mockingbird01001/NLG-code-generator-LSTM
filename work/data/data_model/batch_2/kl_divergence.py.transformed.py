
import tensorflow.compat.v1 as tf
def symmetric_kl_divergence(predicted, actual):
  epsilon = tf.constant(1e-7, dtype=tf.float32, name='epsilon')
  p = tf.math.maximum(predicted, epsilon)
  q = tf.math.maximum(actual, epsilon)
  kld_1 = tf.math.reduce_sum(
      tf.math.multiply(p, tf.math.log(tf.math.divide(p, q))))
  kld_2 = tf.math.reduce_sum(
      tf.math.multiply(q, tf.math.log(tf.math.divide(q, p))))
  return tf.add(kld_1, kld_2)
