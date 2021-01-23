import tensorflow as tf
from kerastuner.tuners import RandomSearch

from utils import *
from model import *
from train import *

from absl import flags
from absl import app

tf.get_logger().setLevel('ERROR')

FLAGS = flags.FLAGS
flags.DEFINE_integer('tv', None, 'version of training dataset')
flags.DEFINE_integer('vv', None, 'version of validation dataset')
flags.DEFINE_integer('bs', 32, 'batch size')

def main(args):
  tv = FLAGS.tv
  vv = FLAGS.vv
  bs = FLAGS.bs

  project_name = f'tv{tv}-vv{vv}-bs{bs}'
  print(f'Project Name: {project_name}')
  print()
  tuner = RandomSearch(
    build_hyper_conv_estimator,
    objective='val_loss',
    max_trials=20,
    executions_per_trial=3,
    directory='hyper_search',
    project_name=project_name,
  )
  
  batch_size = 64
  batches = 4000
  workers = 2
  verbose = 2
  
  tuner.search_space_summary()
  dataset = TFSeqRandomDataGenerator(batch_size, batches)
  valid_dataset = TFSeqRandomDataGenerator(batch_size, 4000, version=1)
  tuner.search(dataset,
               validation_data=valid_dataset,
               epochs=10,
               workers=workers,
               use_multiprocessing=True,
               verbose=verbose)

if __name__ == '__main__':
  app.run(main)
