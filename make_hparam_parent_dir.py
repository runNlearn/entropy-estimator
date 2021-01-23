import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([64, 128, 256, 512]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'rmsprop']))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-2, 1e-3, 1e-4]))
HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([1, 3, 5, 7, 9, 11, 13, 15]))
METRIC_LOSS = 'loss'
METRIC_MAE = 'mean_absolute_error'
PATH_TB_HPARAM = 'tensorboard/hparams'

with tf.summary.create_file_writer(PATH_TB_HPARAM).as_default():
  hp.hparams_config(
    hparams=[HP_BATCH_SIZE, HP_OPTIMIZER, HP_LEARNING_RATE, HP_NUM_LAYERS],
    metrics=[hp.Metric(METRIC_LOSS, display_name='Entropy Loss'),
                 hp.Metric(METRIC_MAE, display_name='MAE')],
  )
