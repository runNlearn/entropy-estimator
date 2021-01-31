import os
import tensorflow as tf

from model import *

path = 'saved_model/bilstmsoft-l2-tv0-vv1-ep100-sgd-bs64-lr0.0001'
config_file = os.path.join(path, 'config.yaml')
config = tf.io.gfile.GFile(config_file, 'r').read()

model = tf.keras.models.model_from_yaml(config)

ckpt = tf.train.Checkpoint(model)
mgr = tf.train.CheckpointManager(ckpt, path, 3)
# It seems `tensorflow-macos` is unstable with tf.train.Checkpoint.
ckpt.restore(mgr.latest_checkpoint).assert_consumed()

print(model.trainable_variables)
#model.save_weights('test_ckpt.h5')
