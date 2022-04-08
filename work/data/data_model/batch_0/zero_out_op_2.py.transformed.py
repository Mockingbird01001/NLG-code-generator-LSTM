
import os.path
import tensorflow as tf
_zero_out_module = tf.load_op_library(
    os.path.join(tf.compat.v1.resource_loader.get_data_files_path(),
                 'zero_out_op_kernel_2.so'))
zero_out = _zero_out_module.zero_out
zero_out2 = _zero_out_module.zero_out2
zero_out3 = _zero_out_module.zero_out3
