
from tensorflow.python.framework import ops
from tensorflow.python.ops.gen_bitwise_ops import *
ops.NotDifferentiable("BitwiseAnd")
ops.NotDifferentiable("BitwiseOr")
ops.NotDifferentiable("BitwiseXor")
ops.NotDifferentiable("Invert")
ops.NotDifferentiable("PopulationCount")
ops.NotDifferentiable("LeftShift")
ops.NotDifferentiable("RightShift")
