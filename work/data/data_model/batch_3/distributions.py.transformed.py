
from tensorflow.python.util import deprecation
with deprecation.silence():
  from tensorflow.python.ops.distributions.bernoulli import Bernoulli
  from tensorflow.python.ops.distributions.beta import Beta
  from tensorflow.python.ops.distributions.categorical import Categorical
  from tensorflow.python.ops.distributions.dirichlet import Dirichlet
  from tensorflow.python.ops.distributions.dirichlet_multinomial import DirichletMultinomial
  from tensorflow.python.ops.distributions.distribution import *
  from tensorflow.python.ops.distributions.exponential import Exponential
  from tensorflow.python.ops.distributions.gamma import Gamma
  from tensorflow.python.ops.distributions.kullback_leibler import *
  from tensorflow.python.ops.distributions.laplace import Laplace
  from tensorflow.python.ops.distributions.multinomial import Multinomial
  from tensorflow.python.ops.distributions.normal import Normal
  from tensorflow.python.ops.distributions.student_t import StudentT
  from tensorflow.python.ops.distributions.uniform import Uniform
del deprecation
