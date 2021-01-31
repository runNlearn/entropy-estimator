import os
import tensorflow as tf

from model import *

# It seems `tensorflow-macos` is unstable with tf.train.Checkpoint.

base_path = 'saved_model/bilstmsoft-l2-tv0-vv1-ep100-sgd-bs64-lr0.0001'
saved_model_paths = tf.io.gfile.glob(os.path.join(base_path, '*.h5'))
saved_model_paths = sorted(saved_model_paths)

target_model_path = saved_model_paths[-1]

loaded_model = tf.keras.models.load_model(target_model_path, compile=False)

print(loaded_model.trainable_variables)
