import tensorflow as tf
from tensorflow import keras
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow.math import exp, log
from tensorflow.experimental import numpy as tnp

class Tweedie(distribution.AutoCompositeTensorDistribution):
  """Tweedie
  """

  def __init__(self,
               loc,
               scale,
               var_power=1.,
               quasi=False,
               a=1.01,
               b=1.99,
               validate_args=False,
               allow_nan_stats=True,
               name='Tweedie'):
    """Construct a (Quasi-)Tweedie Regression with mean `loc`.
    The parameters `loc`must be shaped in a way that supports
    broadcasting.
    Args:
      loc: Floating point tensor; the means of the distribution(s).
      scale: Floating point tensor; the scale of the distribution for Quasi, 
        phi for non-Quasi
      var_power: The variance power, also referred to as "p". The default is 1.
      quasi: Python `bool`, default `False`. When `True` quasi log-liklihood is used.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale], dtype_hint=tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, dtype=dtype, name='loc')
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, dtype=dtype, name='scale')
      self._p = var_power
      self.quasi = quasi
      self.a = a
      self.b = b
      super(Tweedie, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def loc(self):
    """Parameter for the mean."""
    return self._loc
  
  @property
  def scale(self):
    """Parameter for standard deviation."""
    return self._scale

  @property
  def p(self):
    """Parameter for power."""
    return self._p

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])
    
  def _sample_n(self, n, seed=None):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    shape = ps.concat([[n], self._batch_shape_tensor(loc=loc, scale=scale)],
                      axis=0)
    sampled = samplers.normal(
        shape=shape, mean=0., stddev=1., dtype=self.dtype, seed=seed)
    return sampled + loc


  def _log_prob(self, x):
    """Used for the loss of the model -- not an actual log prob"""
    if self.quasi: # from https://www.statsmodels.org/stable/_modules/statsmodels/genmod/families/family.html#Tweedie
      llf = log(2 * tnp.pi * self.scale) + self.p * log(x)
      llf /= -2
      u = (x ** (2 - self.p) - (2 - self.p) * x * self.loc ** (1 - self.p) + (1 - self.p) * self.loc ** (2 - self.p))
      u *= 1 / (self.scale * (1 - self.p) * (2 - self.p))
      return llf - u
    
    else: 
      # from https://github.com/cran/statmod/blob/master/R/tweedie.R negative deviance residuals
      # x1 = x + 0.1 * tf.cast(tf.equal(x, 0), tf.float32)
      # theta = (tf.pow(x1, 1 - self.p) - tf.pow(self.loc, 1 - self.p)) / (1 - self.p)
      # kappa = (tf.pow(x, 2 - self.p) - tf.pow(self.loc, 2 - self.p)) / (2 - self.p)
      # return - 2 * (x * theta - kappa)
      # from https://github.com/cran/mgcv/blob/aff4560d187dfd7d98c7bd367f5a0076faf129b7/R/gamlss.r#L2474
      ethi = tf.exp(-self.p) # assuming p > 0
      p = (self.b + self.a * ethi)/(1+ethi)
      x1 = x + tf.cast(x == 0, tf.float32)
      theta = (tf.pow(x1, 1 - p) - tf.pow(self.loc, 1 - p)) / (1 - p)
      kappa = (tf.pow(x, 2 - p) - tf.pow(self.loc, 2 - p)) / (2 - p)
      return tf.sign(x - self.loc) * tf.sqrt(tf.nn.relu(2 * (x * theta - kappa) * 1 / self.scale))
      


  def _mean(self):
    return self.loc * tf.ones_like(self.scale)
    
  def _stddev(self):
    if self.quasi:
      return self._scale
    else:
      return tf.sqrt(self._scale * tf.pow(self.loc, self.p))

  def _default_event_space_bijector(self):
    return identity_bijector.Identity(validate_args=self.validate_args)

  @classmethod
  def _maximum_likelihood_parameters(cls, value):
    return {'loc': tf.reduce_mean(value, axis=0),
            'scale': tf.math.reduce_std(value, axis=0)}

  def _parameter_control_dependencies(self, is_init):
    assertions = []

    if is_init:
      try:
        self._batch_shape()
      except ValueError:
        raise ValueError(
            'Arguments `loc` and `scale` must have compatible shapes; '
            'loc.shape={}, scale.shape={}.'.format(
                self.loc.shape, self.scale.shape))
      # We don't bother checking the shapes in the dynamic case because
      # all member functions access both arguments anyway.

    if not self.validate_args:
      assert not assertions  # Should never happen.
      return []

    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(assert_util.assert_positive(
          self.scale, message='Argument `scale` must be positive.'))

    return assertions
  
