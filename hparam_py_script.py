import sys
import json
import subprocess

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('tv', 0, 'Version of training set')
flags.DEFINE_integer('vv', 1, 'Version of validation set')

HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([64, 128, 256, 512]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'rmsprop']))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-2, 1e-3, 1e-4]))
HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([1, 3, 5, 7, 9, 11, 13, 15]))
METRIC_LOSS = 'loss'
METRIC_MAE = 'mean_absolute_error'

BASE_CMD = ['python', 'main.py',
            '--data_gen_mode', 'gen',
            '--epochs', '20',
            '--verbose', '0',]

def main(args):
  del args

  tv, vv = FLAGS.tv, FLAGS.vv

  data_version = f'tv{tv}-vv{vv}'
  log_dir = 'tensorboard/hparam'
  with tf.summary.create_file_writer(log_dir).as_default():
    hp.hparams_config(
      hparams=[HP_BATCH_SIZE, HP_OPTIMIZER, HP_LEARNING_RATE, HP_NUM_LAYERS],
      metrics=[hp.Metric(METRIC_LOSS, display_name='Entropy Loss'),
               hp.Metric(METRIC_MAE, display_name='MAE')],
    )

  for num_layers in HP_NUM_LAYERS.domain.values:
    for batch_size in HP_BATCH_SIZE.domain.values:
      for optimizer in HP_OPTIMIZER.domain.values:
        for learning_rate in HP_LEARNING_RATE.domain.values:
          trial_version = f'bs{batch_size}-{optimizer}-lr{learning_rate}-nl{num_layers}'
          hparam_flags = [
            '--num_layers', str(num_layers),
            '--batch_size', str(batch_size),
            '--optimizer', str(optimizer),
            '--learning_rate', str(learning_rate),
            '--tv', str(tv),
            '--vv', str(vv),
          ]
          trial_dir = f'{log_dir}/{data_version}/{trial_version}'
          trial_id = f'{data_version}-{trial_version}'
          output = run(hparam_flags)
          val_loss = output['val_loss']
          val_mae = output['val_mae']
          hparams = {
            'num_layers': num_layers,
            'batch_size': batch_size,
            'optimizer' : optimizer,
            'learning_rate': learning_rate,
          }
          with tf.summary.create_file_writer(trial_dir).as_default():
            hp.hparams(hparams, trial_id)
            tf.summary.scalar(METRIC_LOSS, val_loss, step=1)
            tf.summary.scalar(METRIC_MAE, val_mae, step=1)

def run(hparam_flags):
  cmd = BASE_CMD + hparam_flags
  proc = subprocess.Popen(cmd,
                          stdout=subprocess.PIPE,
                          stderr=sys.stderr)
  json_out = proc.communicate()[0]
  output = json.loads(json_out)
  return output 
  

if __name__ == '__main__':
  app.run(main)
