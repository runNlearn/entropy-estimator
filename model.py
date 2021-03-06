import tensorflow as tf


def ConvEntropyEstimator(num_layers=1, sort=False):
  def fc_block(units, activation='relu'):
    block = tf.keras.Sequential([
      tf.keras.layers.Dense(units),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation(activation)
    ])
    return block
  input = tf.keras.Input(shape=(8, 8), name='block')
  x = tf.keras.layers.Reshape((64,))(input)
  if sort:
    x = tf.keras.layers.Lambda(lambda x: tf.sort(x, axis=-1))(x)
  x = tf.keras.layers.Reshape((8, 8, 1))(x)
  x = tf.keras.layers.Conv2D(256, kernel_size=(8, 8))(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.Flatten()(x)
  for i in range(num_layers):
    x = fc_block(32)(x)
  output = tf.keras.layers.Dense(1, activation='linear', name='entropy')(x)
  return tf.keras.Model(input, output, name='block_entropy_estimator')


def BiLSTMEntropyEstimator(num_layers=1, sort=False):
  def fc_block(units, activation='relu'):
    block = tf.keras.Sequential([
      tf.keras.layers.Dense(units),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation(activation)
    ])
    return block
  input = tf.keras.Input(shape=(8, 8), name='block')
  x = tf.keras.layers.Reshape((64,))(input)
  if sort:
    x = tf.keras.layers.Lambda(lambda x: tf.sort(x, axis=-1))(x)
  x = tf.keras.layers.Reshape((1, 64))(x)
  x = BidirectionalLSTM(256, merge_mode='sum')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  for i in range(num_layers):
    x = fc_block(256)(x)
  output = tf.keras.layers.Dense(1, activation='linear', name='entropy')(x)
  return tf.keras.Model(input, output, name='block_entropy_estimator')


def BiLSTMSoftmax(num_layers=1, sort=False):
  def fc_block(units, activation='sigmoid'):
    block = tf.keras.Sequential([
      tf.keras.layers.Dense(units),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation(activation),
    ])
    return block
  input = tf.keras.Input(shape=(8, 8), name='block')
  x = tf.keras.layers.Reshape((64,))(input)
  if sort:
    x = SortLayer()(x)
  x = tf.keras.layers.Reshape((1, 64))(x)
  x = BidirectionalLSTM(256, merge_mode='sum')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  for i in range(num_layers-1):
    if i == num_layers - 2:
      x = fc_block(256, 'linear')(x)
    else:
      x = fc_block(256)(x)
  x = QuasiEntropyLayer()(x)
  output = x
  return tf.keras.Model(input, output, name='block_entropy_estimator')


def ConvSoftmax(num_layers=1, sort=False):
  def fc_block(units, activation='sigmoid'):
    block = tf.keras.Sequential([
      tf.keras.layers.Dense(units),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation(activation)
    ])
    return block
  input = tf.keras.Input(shape=(8, 8), name='block')
  x = tf.keras.layers.Reshape((64,))(input)
  if sort:
    x = SortLayer()(x)
  x = tf.keras.layers.Reshape((8, 8, 1))(x)
  x = tf.keras.layers.Conv2D(256, (8, 8))(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.Flatten()(x)
  for i in range(num_layers-1):
    if i == num_layers - 2:
      x = fc_block(256, 'linear')(x)
    else:
      x = fc_block(256)(x)
  x = QuasiEntropyLayer()(x)
  output = x
  return tf.keras.Model(input, output, name='block_entropy_estimator')


@tf.keras.utils.register_keras_serializable()
class BidirectionalLSTM(tf.keras.layers.Layer):
  def __init__(self, units, merge_mode='sum', name='bilstm', **kwargs):
    super(BidirectionalLSTM, self).__init__(name=name, **kwargs)
    self.units = units
    self.merge_mode = merge_mode
    self.forward = tf.keras.layers.LSTM(self.units, name='forward')
    self.backward = tf.keras.layers.LSTM(self.units,
                                         go_backwards=True,
                                         name='backward')

  def call(self, inputs):
    x1 = self.forward(inputs)
    x2 = self.backward(inputs)
    if self.merge_mode == 'sum':
      return x1 + x2
    elif self.merge_mode == 'concat':
      return tf.keras.layers.concatenate([x1, x2], axis=-1)

  def get_config(self):
    config = super(BidirectionalLSTM, self).get_config()
    custom_args = {'units': self.units, 'merge_mode': self.merge_mode}
    config.update(custom_args)
    return config
  

@tf.keras.utils.register_keras_serializable()
class SortLayer(tf.keras.layers.Layer):
  def __init__(self, name='sort', **kwargs):
    super(SortLayer, self).__init__(name=name, **kwargs)

  def call(self, inputs):
    return tf.sort(inputs, axis=-1)

  def get_config(self):
    config = super(SortLayer, self).get_config()
    # Add custom arguments.
    # ex) config.update({"units": self.units})
    return config

  # There is no need to define 'from_config' here, since returning
  # `cls(**config)` is the default behavior.
  @classmethod
  def from_config(cls, config):
    return cls(**config)


@tf.keras.utils.register_keras_serializable()
class QuasiEntropyLayer(tf.keras.layers.Layer):
  def __init__(self, name='quasi_entropy', **kwargs):
    super(QuasiEntropyLayer, self).__init__(name=name, **kwargs)
  
  def call(self, inputs):
    prob = tf.math.softmax(inputs, axis=-1) 
    entropy = -tf.reduce_sum(tf.math.log(prob) * prob, -1)
    return entropy

  def get_config(self):
    config = super(QuasiEntropyLayer, self).get_config()
    return config
