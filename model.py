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
  x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256))(x)
  x = tf.keras.layers.BatchNormalization()(x)
  for i in range(num_layers):
    x = fc_block(256)(x)
  output = tf.keras.layers.Dense(1, activation='linear', name='entropy')(x)
  return tf.keras.Model(input, output, name='block_entropy_estimator')


def build_hyper_conv_estimator(hp):
  def fc_block(units, activation='relu'):
    block = tf.keras.Sequential([
      tf.keras.layers.Dense(units),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation(activation)
    ])
    return block
  input = tf.keras.Input(shape=(8, 8), name='block')
  x = tf.keras.layers.Reshape((8, 8, 1))(input)
  x = tf.keras.layers.Conv2D(hp.Int('filter',
                                    min_value=32,
                                    max_value=512,
                                    step=32),
                             kernel_size=(8, 8))(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.Flatten()(x)
  for i in range(hp.Int('num_layers', 2, 20)):
    x = fc_block(units=hp.Int(f'fc_units_{i}',
                              min_value=32,
                              max_value=256,
                              step=32,))(x)
  output = tf.keras.layers.Dense(1, activation='linear', name='entropy')(x)
  model = tf.keras.Model(input, output, name='block_entropy_estimator')
  model.compile(
    optimizer=tf.keras.optimizers.Adam(
      hp.Choice('learning_rate',
                values=[1e-2, 1e-3, 1e-4])),
    loss='huber',
    metrics=['mean_absolute_error']
  )
  return model


def MDN(units, num_components):
  mixture_normal_layer = tfp.layers.MixtureNormal(num_components, name='mixture_normal_layer')
  params_size = mixture_normal_layer.params_size(num_components)
  model = tf.keras.Sequential([
    tf.keras.layers.Reshape(target_shape=(64,)),
    tf.keras.layers.Dense(units, activation='relu'),
    tf.keras.layers.Dense(params_size, activation='linear'),
    mixture_normal_layer])
  
  return model
