import os
import sys
import time
import json
import functools
import subprocess

from absl import app
from absl import logging
from absl import flags

from tf_utils import *
from np_utils import *
from model import *
from train import *

logging.set_verbosity(logging.INFO)
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
flags.DEFINE_integer('epochs', 100, 'Number of epochs.')
flags.DEFINE_integer('batches', 8000, 'Data scale multiplier to number of data.')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('workers', 4, 'Number of workers.')
flags.DEFINE_integer('verbose', 2, 'level of verbose')
flags.DEFINE_integer('tb_freq', 1000, 'update frequency(batches) of tensorboard')
flags.DEFINE_integer('num_layers', 1, 'Number of layers')
flags.DEFINE_integer('tv', 0, 'Training data version')
flags.DEFINE_integer('vv', 1, 'Validation data version')
flags.DEFINE_integer('gpu', -1, 'GPU id, -1 means no gpu')
flags.DEFINE_enum('data_gen_mode', 'gen', ['aot', 'gen', 'tfdata'], 'Data generation mode.')
flags.DEFINE_enum('model', 'conv', ['conv', 'bilstm', 'bilstmsoft', 'convsoft'], 'Type of an entropy estimator.')
flags.DEFINE_enum('optimizer', 'sgd', ['sgd', 'adam', 'rmsprop'], 'Optimizer')
flags.DEFINE_boolean('show_warning', False, 'show Tensorflow warning or not')
flags.DEFINE_boolean('save', False, 'save training history and checkpoints')
flags.DEFINE_boolean('sort', False, 'Sort elements of the block.')

BEST_VALUE = None
MONITORING_VALUE = 'val_loss'

def save_model(ckpt_mgr, model, save_path):
  if not tf.io.gfile.exists(save_path):
    tf.io.gfile.makedirs(save_path)
    file_path = os.path.join(save_path, 'model_config.yaml')
    with tf.io.gfile.GFile(file_path, 'w') as f:
      model.to_yaml(stream=f, indent=4)
  ckpt_mgr.save()


def callback_save_model(epoch, log, ckpt_mgr, model, save_path):
  global BEST_VALUE
  current_value = log[MONITORING_VALUE]
  if BEST_VALUE is None or BEST_VALUE > current_value:
    save_model(ckpt_mgr, model, save_path)
    tf.print(f'Checkpoint is saved.. {current_value}')
  BEST_VALUE = current_value


def main(args):
  del args

  logging.info('Program is started')

  # Check availabe gpu list

  if FLAGS.gpu == -1:
    gpu_setting = '-1'
  else:
    gpu_check_cmd = """nvidia-smi -L | grep Device | awk -F" |:|\)" '{print $6, $10}'"""
    proc = subprocess.Popen(gpu_check_cmd, shell=True, stdout=subprocess.PIPE)
    gpu_list = proc.communicate()[0].decode('utf-8').strip()
    gpu_map = dict(gpu.split() for gpu in gpu_list.split('\n'))
    gpu_setting = gpu_map[str(FLAGS.gpu)]
  os.environ['CUDA_VISIBLE_DEVICES'] = gpu_setting


  if not FLAGS.show_warning:
    tf.get_logger().setLevel('ERROR')

  model_name = FLAGS.model
  batches = FLAGS.batches
  epochs = FLAGS.epochs
  batch_size = FLAGS.batch_size
  data_gen_mode = FLAGS.data_gen_mode
  verbose = FLAGS.verbose
  workers = FLAGS.workers
  optimizer = FLAGS.optimizer
  num_layers = FLAGS.num_layers
  learning_rate = FLAGS.learning_rate
  sort = FLAGS.sort
  tv = FLAGS.tv
  vv = FLAGS.vv

  train_version = '{}-l{}-tv{}-vv{}-ep{}-{}-lr{}-bs{}'.format(
                    model_name, num_layers, tv, vv, epochs,
                    optimizer, batch_size, learning_rate)

  if model_name == 'conv':
    estimator = ConvEntropyEstimator(num_layers, sort)
  elif model_name == 'bilstm':
    estimator = BiLSTMEntropyEstimator(num_layers, sort)
  elif model_name == 'bilstmsoft':
    estimator = BiLSTMSoftmax(num_layers, sort)
  elif model_name == 'convsoft':
    estimator = ConvSoftmax(num_layers, sort)
    

  callbacks = []
  if FLAGS.save:
    tb_path = f'tensorboard/scalars/{train_version}'
    save_path = f'saved_model/{train_version}'
    tb_callback = tf.keras.callbacks.TensorBoard(
      log_dir=tb_path,
      histogram_freq=0,
      write_graph=False,
      write_images=False,
      update_freq=FLAGS.tb_freq,
      profile_batch=0,
      embeddings_freq=0,
      embeddings_metadata=None)
    ckpt = tf.train.Checkpoint(model=estimator)
    ckpt_mgr = tf.train.CheckpointManager(
      checkpoint=ckpt,
      directory=save_path,
      max_to_keep=5)
    ckpt_callback = tf.keras.callbacks.LambdaCallback(
      on_epoch_end=functools.partial(callback_save_model,
                             ckpt_mgr=ckpt_mgr,
                             model=estimator,
                             save_path=save_path))
    callbacks = [tb_callback, ckpt_callback]

  trainer = Trainer(estimator)

  if optimizer == 'sgd':
    optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True)
  elif optimizer == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate)
  elif optimizer == 'rmsprop':
    optimizer = tf.keras.optimizers.RMSprop(learning_rate)

  trainer.compile(optimizer,
                  loss=tf.keras.losses.Huber(),
                  metrics=tf.keras.metrics.MeanAbsoluteError())
  

  
  if data_gen_mode == 'aot':
    data_generator = AOTRandomDataGenerator(batch_size,
                                            batches,
                                            num_process=workers,
                                            context='spawn')
  
    for epoch in range(epochs):
      x, y = data_generator.get_data()
      history = trainer.fit(x, y,
                  batch_size=batch_size,
                  initial_epoch=epoch,
                  epochs=epoch+1,
                  verbose=verbose)

  elif data_gen_mode == 'gen':
    if workers == 1: 
      dataset = get_batch_random_sample_data(batch_size, version=tv)
      valid_dataset = get_batch_random_sample_data(batch_size, version=vv)
      st = time.perf_counter()
      history = trainer.fit(dataset,
                            validation_data=valid_dataset,
                            epochs=epochs,
                            steps_per_epoch=batches,
                            validation_steps=2000,
                            callbacks=callbacks,
                            verbose=verbose)
      et = time.perf_counter()
    else:
      dataset = mp_get_batch_random_sample_data(batch_size, workers, version=tv)
      valid_dataset = mp_get_batch_random_sample_data(batch_size, workers, version=vv)
      st = time.perf_counter()
      history = trainer.fit(dataset,
                            validation_data=valid_dataset,
                            epochs=epochs,
                            steps_per_epoch=batches,
                            validation_steps=2000,
                            callbacks=callbacks,
                            verbose=verbose)
      et = time.perf_counter()

  # There is a memory leak.
  # It seems like tf.data.Dataset is created for every epochs without
  # termination of previous ones. This cumulates unused memories.
  elif data_gen_mode == 'tfdata':
    if workers == 1: 
      dataset = tf_dataset(batch_size)
      valid_dataset = tf_dataset(batch_size)
    else:
      dataset = mp_tf_dataset(batch_size, workers)
      valid_dataset = mp_tf_dataset(batch_size, workers)
    dataset = dataset.take(-1).prefetch(-1)
    valid_dataset = dataset.take(-1).prefetch(-1)
    st = time.perf_counter()
    history = trainer.fit(dataset,
                          validation_data=valid_dataset,
                          epochs=epochs,
                          steps_per_epoch=batches,
                          validation_steps=2000,
                          verbose=verbose)
    et = time.perf_counter()

  if verbose == 0:
    logging.info(f'Epochs {epochs} {et - st:.2f}')
    val_loss = history.history['val_loss'][-1]
    val_mae = history.history['val_mean_absolute_error'][-1]

    result = {'val_loss': val_loss, 'val_mae': val_mae}
    print(json.dumps(result), end='')

if __name__ == '__main__':
  try:
    app.run(main)
  except KeyboardInterrupt:
    logging.info('User stopped program.')
  finally:
    logging.info('Program is terminated.')
    
    
