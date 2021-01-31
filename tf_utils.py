import tensorflow as tf
from np_utils import *

def tf_dataset(batch_size, version=0):
  def make_gen():
    gen = RandomDataGenerator(version)
    while True:
      x, y = gen.batch_random_sample_data(batch_size)
      yield x, y

  return Dataset.from_generator(
            make_gen,
            output_signature=(TensorSpec((batch_size, 8, 8), 'float32'),
                              TensorSpec((batch_size,), 'float32')))


def mp_tf_dataset(batch_size, workers, seeds=None, version=0):
  if seeds is None:
    seeds = np.arange(workers)
  return tf.data.Dataset.from_generator(
      mp_get_batch_random_sample_data,
      output_types=(tf.float32, tf.float32),
      output_shapes=(tf.TensorShape((batch_size, 8, 8)),
                     tf.TensorShape((batch_size,))),
      args=(batch_size, workers, seeds, version))


def tf_calc_entropy(x):
  x = tf.reshape(x, (64,))
  x = tf.cast(x, 'int32')
  _, _, count = tf.unique_with_counts(x)
  nums = tf.reduce_sum(count)
  count = tf.cast(count, 'float32')
  nums = tf.cast(nums, 'float32')
  probs = count / nums
  log_probs = tf.math.log(probs)
  entropy = -tf.reduce_sum(log_probs * probs)
  return entropy
