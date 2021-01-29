import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def _save_config_file(path):
  from model import BiLSTMSoftmax

  test_model = BiLSTMSoftmax(2, True)
  config_yaml = test_model.to_yaml()
  tf.io.gfile.GFile(path, 'w').write(config_yaml)


def test_using_config():
  from model import SortLayer
  from model import QuasiEntropyLayer

  tf.keras.backend.clear_session()
  tmp_file = 'test_config.yaml'

  _save_config_file(tmp_file)

  '''
  If custom layer is decorated by tf.keras.utils.register_keras_serializable(),
  there is no need to load the saved model under the `custom_object_scope`.
      
  custom_objects = {
    'SortLayer': SortLayer,
    'QuasiEntropyLayer': QuasiEntropyLayer,
  }
  with tf.keras.utils.custom_object_scope(custom_objects):
    new_model = tf.keras.models.from_yaml(config_yaml)
  '''

  config_yaml = tf.io.gfile.GFile(tmp_file, 'r').read()
  new_model = tf.keras.models.model_from_yaml(config_yaml)
  new_model.summary()
#  tf.io.gfile.remove(tmp_file)


def test_using_pb():
  from model import BiLSTMSoftmax

  tf.keras.backend.clear_session()
  path_tmp_dir = 'testing'
  test_model = BiLSTMSoftmax(2, True)
  test_model.save(path_tmp_dir)
  new_model = tf.keras.models.load_model(path_tmp_dir, compile=False)
  new_model.summary()
  tf.io.gfile.rmtree(path_tmp_dir)

if __name__ == '__main__':
#  try:
    test_using_config()
#  except:
#    print('`test_using_config` is failed')
#
#  try:
#    test_using_pb()
#  except:
#    print('`test_using_pb` is failed')
