from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_compression as tfc

interpolate = 1 # 1:NearestNeighbor

class ChannelNorm(tf.keras.layers.Layer):
  """Implement ChannelNorm.
  Based on this paper and keras' InstanceNorm layer:
    Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton.
    "Layer normalization."
    arXiv preprint arXiv:1607.06450 (2016).
  """

  def __init__(self,
               epsilon: float = 1e-3,
               center: bool = True,
               scale: bool = True,
               beta_initializer="zeros",
               gamma_initializer="ones",
               **kwargs):
    """Instantiate layer.
    Args:
      epsilon: For stability when normalizing.
      center: Whether to create and use a {beta}.
      scale: Whether to create and use a {gamma}.
      beta_initializer: Initializer for beta.
      gamma_initializer: Initializer for gamma.
      **kwargs: Passed to keras.
    """
    super(ChannelNorm, self).__init__(**kwargs)

    self.axis = -1
    self.epsilon = epsilon
    self.center = center
    self.scale = scale
    self.beta_initializer = tf.keras.initializers.get(beta_initializer)
    self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)

  def build(self, input_shape):
    self._add_gamma_weight(input_shape)
    self._add_beta_weight(input_shape)
    self.built = True
    super().build(input_shape)

  def call(self, inputs, modulation=None):
    mean, variance = self._get_moments(inputs)
    # inputs = tf.Print(inputs, [mean, variance, self.beta, self.gamma], "NORM")
    return tf.nn.batch_normalization(
        inputs, mean, variance, self.beta, self.gamma, self.epsilon,
        name="normalize")

  def _get_moments(self, inputs):
    # Like tf.nn.moments but unbiased sample std. deviation.
    # Reduce over channels only.
    mean = tf.reduce_mean(inputs, [self.axis], keepdims=True, name="mean")
    variance = tf.reduce_sum(
        tf.squared_difference(inputs, tf.stop_gradient(mean)),
        [self.axis], keepdims=True, name="variance_sum")
    # Divide by N-1
    inputs_shape = tf.shape(inputs)
    counts = tf.reduce_prod([inputs_shape[ax] for ax in [self.axis]])
    variance /= (tf.cast(counts, tf.float32) - 1)
    return mean, variance

  def _add_gamma_weight(self, input_shape):
    dim = input_shape[self.axis]
    shape = (dim,)

    if self.scale:
      self.gamma = self.add_weight(
          shape=shape,
          name="gamma",
          initializer=self.gamma_initializer)
    else:
      self.gamma = None

  def _add_beta_weight(self, input_shape):
    dim = input_shape[self.axis]
    shape = (dim,)

    if self.center:
      self.beta = self.add_weight(
          shape=shape,
          name="beta",
          initializer=self.beta_initializer)
    else:
      self.beta = None


def MV_analysis(tensor, num_filters, M):
  """Builds the analysis transform."""

  with tf.variable_scope("MV_analysis"):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=None)
      layer2 = ChannelNorm()
      layer3 = tf.keras.layers.ReLU()
      tensor = layer3(layer2(layer(tensor)))

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=None)
      layer2 = ChannelNorm()
      layer3 = tf.keras.layers.ReLU()
      tensor = layer3(layer2(layer(tensor)))

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=None)
      layer2 = ChannelNorm()
      layer3 = tf.keras.layers.ReLU()
      tensor = layer3(layer2(layer(tensor)))

    with tf.variable_scope("layer_3"):
      layer = tfc.SignalConv2D(
          M, (3, 3), corr=True, strides_down=2, padding="same_zeros",
          use_bias=False, activation=None)
      tensor = layer(tensor)

    return tensor


def MV_synthesis(tensor, num_filters):
  """Builds the synthesis transform."""

  with tf.variable_scope("MV_synthesis"):
    with tf.variable_scope("layer_0"):

      tensor = tf.image.resize_images(tensor, [2 * tf.shape(tensor)[1], 2 * tf.shape(tensor)[2]], method=interpolate)
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=False, strides_up=1, padding="same_zeros",
          use_bias=True, activation=None)
      layer2 = ChannelNorm()
      layer3 = tf.keras.layers.ReLU()
      tensor = layer3(layer2(layer(tensor)))

    with tf.variable_scope("layer_1"):
      tensor = tf.image.resize_images(tensor, [2 * tf.shape(tensor)[1], 2 * tf.shape(tensor)[2]], method=interpolate)
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=False, strides_up=1, padding="same_zeros",
          use_bias=True, activation=None)
      layer2 = ChannelNorm()
      layer3 = tf.keras.layers.ReLU()
      tensor = layer3(layer2(layer(tensor)))

    with tf.variable_scope("layer_2"):
      tensor = tf.image.resize_images(tensor, [2 * tf.shape(tensor)[1], 2 * tf.shape(tensor)[2]], method=interpolate)
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=False, strides_up=1, padding="same_zeros",
          use_bias=True, activation=None)
      layer2 = ChannelNorm()
      layer3 = tf.keras.layers.ReLU()
      tensor = layer3(layer2(layer(tensor)))

    with tf.variable_scope("layer_3"):
      tensor = tf.image.resize_images(tensor, [2 * tf.shape(tensor)[1], 2 * tf.shape(tensor)[2]], method=interpolate)
      layer = tfc.SignalConv2D(
          2, (3, 3), corr=False, strides_up=1, padding="same_zeros",
          use_bias=True, activation=None)
      tensor = layer(tensor)

    return tensor


def Res_analysis(tensor, num_filters, M, reuse=False):
  """Builds the analysis transform."""

  with tf.variable_scope("analysis", reuse=reuse):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=None)
      layer2 = ChannelNorm()
      layer3 = tf.keras.layers.ReLU()
      tensor = layer3(layer2(layer(tensor)))

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=None)
      layer2 = ChannelNorm()
      layer3 = tf.keras.layers.ReLU()
      tensor = layer3(layer2(layer(tensor)))

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=None)
      layer2 = ChannelNorm()
      layer3 = tf.keras.layers.ReLU()
      tensor = layer3(layer2(layer(tensor)))

    with tf.variable_scope("layer_3"):
      layer = tfc.SignalConv2D(
          M, (3, 3), corr=True, strides_down=2, padding="same_zeros",
          use_bias=False, activation=None)
      tensor = layer(tensor)

    return tensor


def Res_synthesis(tensor, num_filters, reuse=False):
  """Builds the synthesis transform."""

  with tf.variable_scope("synthesis", reuse=reuse):
    with tf.variable_scope("layer_0"):
      tensor = tf.image.resize_images(tensor, [2 * tf.shape(tensor)[1], 2 * tf.shape(tensor)[2]], method=interpolate)
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=False, strides_up=1, padding="same_zeros",
          use_bias=True, activation=None)
      layer2 = ChannelNorm()
      layer3 = tf.keras.layers.ReLU()
      tensor = layer3(layer2(layer(tensor)))

    with tf.variable_scope("layer_1"):
      tensor = tf.image.resize_images(tensor, [2 * tf.shape(tensor)[1], 2 * tf.shape(tensor)[2]], method=interpolate)
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=False, strides_up=1, padding="same_zeros",
          use_bias=True, activation=None)
      layer2 = ChannelNorm()
      layer3 = tf.keras.layers.ReLU()
      tensor = layer3(layer2(layer(tensor)))

    with tf.variable_scope("layer_2"):
      tensor = tf.image.resize_images(tensor, [2 * tf.shape(tensor)[1], 2 * tf.shape(tensor)[2]], method=interpolate)
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=False, strides_up=1, padding="same_zeros",
          use_bias=True, activation=None)
      layer2 = ChannelNorm()
      layer3 = tf.keras.layers.ReLU()
      tensor = layer3(layer2(layer(tensor)))

    with tf.variable_scope("layer_3"):
      tensor = tf.image.resize_images(tensor, [2 * tf.shape(tensor)[1], 2 * tf.shape(tensor)[2]], method=interpolate)
      layer = tfc.SignalConv2D(
          3, (3, 3), corr=False, strides_up=1, padding="same_zeros",
          use_bias=True, activation=None)
      tensor = layer(tensor)

    return tensor