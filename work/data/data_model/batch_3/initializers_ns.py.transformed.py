
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variables as _variables
zeros = init_ops.zeros_initializer
ones = init_ops.ones_initializer
constant = init_ops.constant_initializer
random_uniform = init_ops.random_uniform_initializer
random_normal = init_ops.random_normal_initializer
truncated_normal = init_ops.truncated_normal_initializer
uniform_unit_scaling = init_ops.uniform_unit_scaling_initializer
variance_scaling = init_ops.variance_scaling_initializer
orthogonal = init_ops.orthogonal_initializer
identity = init_ops.identity_initializer
variables = _variables.variables_initializer
global_variables = _variables.global_variables_initializer
local_variables = _variables.local_variables_initializer
del init_ops
del _variables
