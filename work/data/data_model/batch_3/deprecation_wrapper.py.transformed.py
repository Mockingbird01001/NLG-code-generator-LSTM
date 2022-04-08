
"""Compatibility wrapper for TensorFlow modules to support deprecation messages.
Please use module_wrapper instead.
TODO(yifeif): remove once no longer referred by estimator
"""
from tensorflow.python.util import module_wrapper
DeprecationWrapper = module_wrapper.TFModuleWrapper
