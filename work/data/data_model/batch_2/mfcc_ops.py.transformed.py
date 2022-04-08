
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import dct_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('signal.mfccs_from_log_mel_spectrograms')
@dispatch.add_dispatch_support
def mfccs_from_log_mel_spectrograms(log_mel_spectrograms, name=None):
  """Computes [MFCCs][mfcc] of `log_mel_spectrograms`.
  Implemented with GPU-compatible ops and supports gradients.
  [Mel-Frequency Cepstral Coefficient (MFCC)][mfcc] calculation consists of
  taking the DCT-II of a log-magnitude mel-scale spectrogram. [HTK][htk]'s MFCCs
  use a particular scaling of the DCT-II which is almost orthogonal
  normalization. We follow this convention.
  All `num_mel_bins` MFCCs are returned and it is up to the caller to select
  a subset of the MFCCs based on their application. For example, it is typical
  to only use the first few for speech recognition, as this results in
  an approximately pitch-invariant representation of the signal.
  For example:
  ```python
  batch_size, num_samples, sample_rate = 32, 32000, 16000.0
  pcm = tf.random.normal([batch_size, num_samples], dtype=tf.float32)
  stfts = tf.signal.stft(pcm, frame_length=1024, frame_step=256,
                         fft_length=1024)
  spectrograms = tf.abs(stfts)
  num_spectrogram_bins = stfts.shape[-1].value
  lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
    upper_edge_hertz)
  mel_spectrograms = tf.tensordot(
    spectrograms, linear_to_mel_weight_matrix, 1)
  mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
    linear_to_mel_weight_matrix.shape[-1:]))
  log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
  mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
    log_mel_spectrograms)[..., :13]
  ```
  Args:
    log_mel_spectrograms: A `[..., num_mel_bins]` `float32`/`float64` `Tensor`
      of log-magnitude mel-scale spectrograms.
    name: An optional name for the operation.
  Returns:
    A `[..., num_mel_bins]` `float32`/`float64` `Tensor` of the MFCCs of
    `log_mel_spectrograms`.
  Raises:
    ValueError: If `num_mel_bins` is not positive.
  [mfcc]: https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
  [htk]: https://en.wikipedia.org/wiki/HTK_(software)
  """
  with ops.name_scope(name, 'mfccs_from_log_mel_spectrograms',
                      [log_mel_spectrograms]):
    log_mel_spectrograms = ops.convert_to_tensor(log_mel_spectrograms)
    if (log_mel_spectrograms.shape.ndims and
        log_mel_spectrograms.shape.dims[-1].value is not None):
      num_mel_bins = log_mel_spectrograms.shape.dims[-1].value
      if num_mel_bins == 0:
        raise ValueError('num_mel_bins must be positive. Got: %s' %
                         log_mel_spectrograms)
    else:
      num_mel_bins = array_ops.shape(log_mel_spectrograms)[-1]
    dct2 = dct_ops.dct(log_mel_spectrograms, type=2)
    return dct2 * math_ops.rsqrt(
        math_ops.cast(num_mel_bins, dct2.dtype) * 2.0)
