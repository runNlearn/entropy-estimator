import os
import time
import multiprocessing as mp

from absl import app
from absl import logging
from absl import flags

from utils import *
from model import *
from train import *

from tensorboard.plugins.hparams import api as hp

logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS

flags.DEFINE_integer('tv', 0, 'Training data version')
flags.DEFINE_integer('vv', 1, 'Validation data version')

HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([64, 128, 256, 512]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'rmsprop']))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-2, 1e-3, 1e-4]))
HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([1, 3, 5, 7, 9, 11, 13, 15]))
METRIC_LOSS = 'loss'
METRIC_MAE = 'mean_absolute_error'

PATH_TB_HPARAM = 'tensorboard/hparams'
PATH_TB_SCALAR = 'tensorboard/scalars'

def run(hparams, data_version):
  batch_size = hparams[HP_BATCH_SIZE]
  optimizer = hparams[HP_OPTIMIZER]
  learning_rate = hparams[HP_LEARNING_RATE]
  num_layers = hparams[HP_NUM_LAYERS]

  train_version = f'bs{batch_size}-{optimizer}-lr{learning_rate}-nl{num_layers}'
  scalar_trial_dir = f'{PATH_TB_SCALAR}/{data_version}/{train_version}'

  print('Start new trial')
  print(f'  Version {data_version}')
  print(f'  Batch Size: {batch_size}')
  print(f'  Optimizer: {optimizer}')
  print(f'  Learning Rate: {learning_rate}')
  print(f'  Number of layers: {num_layers}')
  print()


  if optimizer == 'sgd':
    optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True)
  elif optimizer == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate)
  elif optimizer == 'rmsprop':
    optimizer = tf.keras.optimizers.RMSprop(learning_rate)

  estimator = ConvEntropyEstimator(num_layers)

  estimator.compile(
    optimizer, 
    loss=tf.keras.losses.Huber(),
    metrics=[tf.keras.losses.MeanAbsoluteError()])

  callbacks = [
    tf.keras.callbacks.TensorBoard(
      log_dir=scalar_trial_dir,
      histogram_freq=0,
      write_graph=False,
      write_images=False,
      update_freq=1000,
      profile_batch=0,
      embeddings_freq=0,
      embeddings_metadata=None),
  ]

  train_dataset = make_tf_data_random_dataset(batch_size)
  valid_dataset = make_tf_data_random_dataset(batch_size)
  train_dataset = train_dataset.prefetch(-1)
  valid_dataset = valid_dataset.prefetch(-1)

  history = estimator.fit(
    train_dataset,
    validation_data=valid_dataset,
    steps_per_epoch=8000,
    validation_steps=2000,
    epochs=10,
    callbacks=callbacks,
    verbose=2)

  val_mae = history.history['val_mean_absolute_error'][-1]
  val_loss = history.history['val_loss'][-1]

  hparam_trial_dir = f'{PATH_TB_HPARAM}/{data_version}-{train_version}'
  with tf.summary.create_file_writer(hparam_trial_dir).as_default():
    hp.hparams(hparams, f'{data_version}-{train_version}')
    tf.summary.scalar(METRIC_LOSS, val_loss, step=1)
    tf.summary.scalar(METRIC_MAE, val_mae, step=1)


def main(args):
  del args

  tf.get_logger().setLevel('ERROR')

  tv, vv = FLAGS.tv, FLAGS.vv 
  
  data_version = f'tv{tv}-vv{vv}'

  for num_layers in HP_NUM_LAYERS.domain.values:
    for batch_size in HP_BATCH_SIZE.domain.values:
      for optimizer in HP_OPTIMIZER.domain.values:
        for learning_rate in HP_LEARNING_RATE.domain.values:
          hparams = {
            HP_NUM_LAYERS: num_layers,
            HP_BATCH_SIZE: batch_size,
            HP_OPTIMIZER: optimizer,
            HP_LEARNING_RATE: learning_rate
          }
          run(hparams, data_version)


if __name__ == '__main__':
  app.run(main)
