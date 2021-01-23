import tensorflow as tf

class Trainer(tf.keras.Model):
  def __init__(self, estimator):
    super(Trainer, self).__init__()
    self.estimator = estimator
    # self.mean_entropy_true = tf.keras.metrics.Mean(name='entropy_true')
    # self.mean_entropy_pred = tf.keras.metrics.Mean(name='entropy_pred')

  def call(self, input):
    return self.estimator(input)
  
  # @property
  # def metrics(self):
  #   metrics = self.compiled_loss.metrics + self.compiled_metrics.metrics
  #   metrics = metrics + [self.mean_entropy_true, self.mean_entropy_pred]
  #   return metrics

  def train_step(self, data):
    block, entropy = data

    with tf.GradientTape() as tape:
      entropy_pred = self(block, training=True)
      loss = self.compiled_loss(entropy, entropy_pred)
    grad = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

    self.compiled_metrics.update_state(entropy, entropy_pred)
    # self.mean_entropy_true.update_state(entropy)
    # self.mean_entropy_pred.update_state(entropy_pred)

    result = {m.name: m.result() for m in self.metrics}
    # result['entropy_true'] = self.mean_entropy_true.result()
    # result['entropy_pred'] = self.mean_entropy_pred.result()
    return result
