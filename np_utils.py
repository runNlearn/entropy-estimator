import os
import sys
import time
import multiprocessing as mp
import numpy as np

from absl import logging

class RandomDataGenerator(object):
  def __init__(self, seed=None, version=0):
    self.random_sample_data = getattr(self, f'_random_sample_data_v{version}')
    self.rng = np.random.default_rng(seed)
    self.version = version

  def _random_sample_data_v0(self):
    v = self.rng.uniform()
    if v <= 0.25:
      a = self.rng.uniform(low=0.0, high=1.0)
      b = self.rng.uniform(low=10, high=100)
    elif v <= 0.50:
      a = self.rng.uniform(low=10, high=100)
      b = self.rng.uniform(low=0.0, high=1.0)
    elif v <= 0.75:
      a = self.rng.uniform(low=0.0, high=1.0)
      b = self.rng.uniform(low=0.0, high=1.0)
    elif v <= 1.0:
      a = self.rng.uniform(low=10, high=100)
      b = self.rng.uniform(low=10, high=100)

    scale = self.rng.uniform()
    block = (self.rng.beta(size=(64,), a=a, b=b) - 0.5) * scale + 0.5
    block = np.round(block *  255) -  128
    entropy = np_calc_entropy(block)
    return block.reshape(8, 8), entropy

  def _random_sample_data_v1(self):
    n_group = int(self.rng.uniform(low=1, high=64))
    value_range = np.arange(256) - 128
    elements = self.rng.choice(value_range,
                               size=(n_group,),
                               replace=False,
                               shuffle=False)
    if n_group == 1:
      block = np.ones((64,), dtype='int32') * elements
      entropy = np_calc_entropy(block)
    else:
      boundaries = np.sort(self.rng.choice(np.arange(1, 63),
                                           size=(n_group - 1),
                                           replace=False,
                                           shuffle=False))
      block = np.zeros((64,), dtype='int32')
      boundaries = np.insert(boundaries, 0, 0)
      boundaries = np.append(boundaries, 63)
      for i in range(len(boundaries) - 1):
        low = boundaries[i]
        high = boundaries[i + 1]
        block[low:high + 1] = elements[i]

      if self.rng.uniform() > 0.5:
        self.rng.shuffle(block)
      entropy = np_calc_entropy(block)
    return block.reshape(8, 8), entropy

  def batch_random_sample_data(self, batch_size):
    x, y = zip(*[self.random_sample_data() for _ in range(batch_size)])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


# TODO: per process random seed
class AOTRandomDataGenerator(RandomDataGenerator):
  def __init__(self,
               batch_size,
               batches,
               version=0,
               num_process=2,
               context='spawn'):
    super(AOTRandomDataGenerator, self).__init__(version=version)
    self.batches = batches
    ctx = mp.get_context(context)
    self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(num_process)])
    fn = lambda: self.batch_random_sample_data(batch_size * batches // num_process)
    ps = [ctx.Process(target=_push_data, args=(work_remote, remote, CloudpickleWrapper(fn)))
          for work_remote, remote in zip(self.work_remotes, self.remotes)]

    for process in ps:
      process.daemon = True
      process.start()
    for work_remote in self.work_remotes:
      work_remote.close()

  def get_data(self):
    logging.info(f'Generating data for {self.batches} steps...')
    st = time.perf_counter()
    xs, ys = [], []
    for remote in self.remotes:
      remote.send('go')
      x, y = remote.recv()
      xs.append(x)
      ys.append(y)
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    et = time.perf_counter()
    logging.info('Data generation took {:.1f} secs.'.format(et - st))
    return x, y


def _push_data(remote, parent_remote, fn_wrapper):
  parent_remote.close()
  try:
    while True:
      x, y = fn_wrapper.x()
      signal = remote.recv()
      remote.send((x, y))
  except KeyboardInterrupt:
    print('Subprocess for data generation: got KeyboardInterrupt')


def np_calc_entropy(x):
  x = np.reshape(x, (64,)).astype('int32')
  count = np.unique(x, return_counts=True)[1]
  nums = np.sum(count)
  probs = (count / nums).astype('float32')
  log_probs = np.log(probs)
  entropy = -np.sum(log_probs * probs)
  return entropy


def np_dct2d(x):
  pass


def get_batch_random_sample_data(batch_size, seed=None, version=0):
  gen = RandomDataGenerator(seed, version)
  while True:
    data = gen.batch_random_sample_data(batch_size)
    yield data


def _put_batch_random_sample_data(batch_size, q, seed=None, version=0):
  gen = RandomDataGenerator(seed, version)
  while True:
    try:
      data = gen.batch_random_sample_data(batch_size)
      q.put(data)
    except KeyboardInterrupt:
      logging.info(f'Subprocess {os.getpid()} is terminated')
      return


def mp_get_batch_random_sample_data(batch_size, workers, seeds=None, version=0):
  ctx = mp.get_context('spawn')
  q = ctx.Queue(workers * 2)
  if seeds is None:
    ps = [ctx.Process(target=_put_batch_random_sample_data,
                      args=(batch_size, q, seed, version))
          for seed in range(workers)]
  else:
    ps = [ctx.Process(target=_put_batch_random_sample_data,
                      args=(batch_size, q, seed, version))
          for seed in seeds]
  for p in ps:
    p.daemon = True
    p.start()

  while True:
    yield q.get()
